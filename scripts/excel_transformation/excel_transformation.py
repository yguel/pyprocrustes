# This script imports data from an Excel file.
# It contains data points with fields: 
# ["Pose #","Robot_X",	"Robot_Y", "Robot_Z", "Robot_Rx (deg)", "Robot_Ry (deg)", "Robot_Rz (deg)", "FARO_X", "FARO_Y", "FARO_Z"]
# If the "Pose #" field contains an integer then the points are used to compute the change of coordinate systems between points Robot_X, Robot_Y, Robot_Z and FARO_X, FARO_Y, FARO_Z.

# If the "Pose #" field contains a string then the points are only entered in one coordinate system either Robot or FARO, and one we want is to compute the coordinates of that point in the one missing.

from pyprocrustes import compute_transformation
import numpy as np
import click
import pandas as pd
import json

def read_excel_file(file_path: str) -> dict:
    """
    Read an Excel file and return the data as a dictionary.

    Parameters
    ----------
    file_path : str
        The path to the Excel file.

    Returns
    -------
    dict
        A dictionary with 3 keys:
            * "Robot Frame" : a list of points in Robot_X, Robot_Y, Robot_Z coordinate systems that are used to compute the coordinate system transformation
            * "Faro Frame" : a list of points in FARO_X, FARO_Y, FARO_Z coordinate systems that are used to compute the coordinate system transformation
            * "to transform" : a list of dict containing points in either Robot or FARO coordinate systems that we want to compute the coordinates of that point in the one missing: the dict as the form { "frame" : "Robot" or "FARO", "points" : [point1, point2, ...], "names" : [name1, name2, ...] }
    """

    # Read the Excel file
    df = pd.read_excel(file_path)

    # Initialize the dictionary to store the data
    data = {
        "Robot Frame": [],
        "Faro Frame": [],
        "to transform": {}
    }
    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        # Check if the "Pose #" field is an integer
        if isinstance(row["Pose #"], int):
            # If it is an integer, add the points to the Robot and Faro frames
            robot_pt = np.array([row["Robot_X"], row["Robot_Y"], row["Robot_Z"]])
            faro_pt = np.array([row["FARO_X"], row["FARO_Y"], row["FARO_Z"]])
            # Append the points to the respective lists
            data["Robot Frame"].append(robot_pt)
            data["Faro Frame"].append(faro_pt)
        else:
            # If it is a string, add the points to the "to transform" list
            name = row["Pose #"]
            # Check if the name is in the Robot or Faro coordinate system
            # if Robot_X, Robot_Y, Robot_Z are not NaN then the point is in the Robot coordinate system
            if not pd.isna(row["Robot_X"]) and not pd.isna(row["Robot_Y"]) and not pd.isna(row["Robot_Z"]):
                robot_pt = np.array([row["Robot_X"], row["Robot_Y"], row["Robot_Z"]])
                # Append the point to the "to transform" list
                if "Robot" not in data["to transform"]:
                    data["to transform"]["Robot"] = {"points": [], "names": []}
                data["to transform"]["Robot"]["points"].append(robot_pt)
                data["to transform"]["Robot"]["names"].append(name)
            elif not pd.isna(row["FARO_X"]) and not pd.isna(row["FARO_Y"]) and not pd.isna(row["FARO_Z"]):
                faro_pt = np.array([row["FARO_X"], row["FARO_Y"], row["FARO_Z"]])
                # Append the point to the "to transform" list
                if "FARO" not in data["to transform"]:
                    data["to transform"]["FARO"] = {"points": [], "names": []}
                data["to transform"]["FARO"]["points"].append(faro_pt)
                data["to transform"]["FARO"]["names"].append(name)    
    return data


@click.command( help="Compute the transformation matrix from an Excel file and write the results to a new Excel file." )
@click.argument( "excel_file_path", type=str, help="The path of the Excel file to read." )
@click.argument( "output_file_path", type=str, help="The name of json file for exporting results" )
def main(excel_file_path: str, output_file_path: str):
    """
    Main function to read the Excel file, compute the transformation and write the results to a new Excel file.

    Parameters
    ----------
    excel_file_name : str
        The name of the Excel file to read.
    output_file_name : str
        The name of the Excel file to write the results to.
    """
    # Read the Excel file
    data = read_excel_file(excel_file_path)

    # Compute the transformation matrix
    Robot_points = np.array(data["Robot Frame"])
    Faro_points = np.array(data["Faro Frame"])
    T_faro2robot, unique = compute_transformation(Faro_points, Robot_points)
    # Check if the transformation is unique
    if not unique:
        print("The transformation is not unique. The points are aligned and many possible transformations exist.")
    T_robot2faro = np.linalg.inv(T_faro2robot)
    # Create a json file with the transformation matrix and the transformed points
    
    #Transform the points
    transformed_points = []
    for frame, points in data["to transform"].items():
        if frame == "Robot":
            # Transform the points from Robot to Faro       
            for k,point in enumerate(points["points"]):
                homogeneous_point = np.append(point, 1)
                transformed_point = T_robot2faro @ homogeneous_point
                pt_data = {
                    "name": points["names"][k],
                    "Faro frame": transformed_point[:3].tolist(),
                    "Robot frame": point.tolist()
                }
                transformed_points.append(pt_data)
        elif frame == "FARO":
            # Transform the points from Faro to Robot
            for k,point in enumerate(points["points"]):
                homogeneous_point = np.append(point, 1)
                transformed_point = T_faro2robot @ homogeneous_point
                pt_data = {
                    "name": points["names"][k],
                    "Robot frame": transformed_point[:3].tolist(),
                    "Faro frame": point.tolist()
                }
                transformed_points.append(pt_data)
    
    report = {
        "Frame transformation matrix Faro -> Robot": T_faro2robot.tolist(),
        "Frame transformation matrix Robot -> Faro": T_robot2faro.tolist(),
        "transformed points": transformed_points
    }
    
    with open(output_file_path, "w") as outfile:
        json.dump(report, outfile, indent=2)

    
if __name__ == "__main__":
    main()