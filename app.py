from flask import Flask, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from io import StringIO
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import numpy as np
import trimesh
import json
import os
import time

app = Flask(__name__)

CORS(app, origins="*")

target_number_of_faces = 100

# Configuration of the database  
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root@localhost:3306/modules3d'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  

db = SQLAlchemy(app)

# The model for file associations
class FileAssociation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    obj_file_path = db.Column(db.String(255), nullable=False)
    image_file_path = db.Column(db.String(255), nullable=False)
    calculation_results = db.Column(db.JSON, nullable=True)

# Function to store objects in the database
def store_object_in_database(object_data):
    try:
        # Create a new FileAssociation instance
        new_file_association = FileAssociation(
            obj_file_path=object_data['obj_file_path'],
            image_file_path=object_data['image_file_path'],
            calculation_results=object_data['calculation_results']
        )

        # Add and commit to the database
        db.session.add(new_file_association)
        db.session.commit()

        # Return the stored object
        return new_file_association
    except Exception as e:
        # Handle exceptions
        print(f"Error storing object in the database: {e}")
        return None

# Function to get an object from the database by ID
def get_object_from_database_by_id(file_id):
    try:
        # Query the database for the FileAssociation with the given ID
        file_association = FileAssociation.query.filter_by(id=file_id).first()

        # Return the retrieved object
        return file_association
    
    except Exception as e:
        # Handle exceptions
        print(f"Error retrieving object from the database: {e}")
        return None

# Route to serve .obj files
@app.route('/get_obj/<filename>')
def get_obj(filename):
    return send_from_directory("C:/Users/emy7u/Downloads/3DPotteryDataset_v_1/3D Models/All Models/", filename)

# Route to serve image files
@app.route('/get_image/<filename>')
def get_image(filename):
    return send_from_directory("C:/Users/emy7u/Downloads/3DPotteryDataset_v_1/Thumbnails/All Models/", filename)

# Route to get details by image ID
@app.route('/get_details_by_id/<int:image_id>', methods=['GET'])
def get_details_by_id(image_id):
    try:
        # Query the database for the FileAssociation with the given ID
        file_association = FileAssociation.query.get(image_id)

        if file_association:
            img_link = url_for('get_image', filename=file_association.image_file_path.split('/')[-1], _external=True)

            # Extract details from the FileAssociation entry
            details = {
                "id": file_association.id,
                "obj_link": file_association.obj_file_path,
                "image_link": img_link,
                "calculation_results" : file_association.calculation_results,
                "message": "data is being send"  # Add any other fields you want
            }

            return jsonify(details)
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        print(f"Error fetching details by ID: {e}")
        return jsonify({"error": "Error fetching details by ID"}), 500

# Route to get all images    
@app.route('/get_images', methods=['GET'])
def get_images():
    try:
        # Query the database for all FileAssociation entries
        file_associations = FileAssociation.query.all()

        # Specify the base URL for get_image
        base_url = 'http://127.0.0.1:5000'

        # Extract image data (ID and URL) from the entries
        image_data = [
            {"id": file_association.id, "image_link": f"{base_url}{url_for('get_image', filename=file_association.image_file_path.split('/')[-1])}",
             "obj_link": file_association.obj_file_path
        
     
             }
            for file_association in file_associations
        ]

        return jsonify(image_data)
    except Exception as e:
        print(f"Error fetching images: {e}")
        return jsonify({"error": "Error fetching images"}), 500

@app.route('/get_results/<image_id>', methods=['GET'])
def get_results_by_id(image_id):
    try:
        file_association = get_object_from_database_by_id(image_id)
        if file_association:
            return jsonify(json.loads(file_association.calculation_results))
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        print(f"Error fetching results by ID: {e}")
        return jsonify({"error": "Error fetching results by ID"}), 500

# Function to remove the image file
def remove_image_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Image file removed: {file_path}")
        else:
            print(f"Image file not found: {file_path}")
    except Exception as e:
        print(f"Error removing image file: {e}")

# Route to remove an image by ID
@app.route('/remove_image/<image_id>', methods=['DELETE'])
def remove_image(image_id):
    try:
        # Find the FileAssociation entry by ID
        file_association = get_object_from_database_by_id(image_id)

        if file_association:
            # Delete the image file (optional)
            remove_image_file(file_association.image_file_path)

            # Delete the FileAssociation entry from the database
            db.session.delete(file_association)
            db.session.commit()

            return jsonify({"message": "Image removed successfully"})
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        print(f"Error removing image: {e}")
        return jsonify({"error": "Error removing image"}), 500

def get_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"Error reading file content: {e}")
        return None

# Route to serve image files by ID
@app.route('/get_image_by_id/<int:image_id>', methods=['GET'])
def get_image_by_id(image_id):
    try:
        # Query the database for the FileAssociation with the given ID
        file_association = FileAssociation.query.get(image_id)

        if file_association:
            # Extract the filename from the FileAssociation entry and serve the image
            filename = file_association.image_file_path.split('/')[-1]
            return send_from_directory("C:/Users/emy7u/Downloads/3DPotteryDataset_v_1/Thumbnails/All Models/", filename)
        else:
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        print(f"Error fetching image by ID: {e}")
        return jsonify({"error": "Error fetching image by ID"}), 500

class CustomObjLoader:
    def __init__(self, file_content):
        self.file_content = file_content
        self.vertices = []
        self.faces = []
        self.x_vals = []
        self.y_vals = []
        self.z_vals = []
        self.read_file()

    def read_file(self):
        for line in self.file_content.splitlines():
            if line.startswith('v'):
                vertex = line.split()
                if len(vertex) == 4:
                    self.vertices.append([float(vertex[1]), float(vertex[2]), float(vertex[3])])
                    self.x_vals.append(float(vertex[1]))
                    self.y_vals.append(float(vertex[2]))
                    self.z_vals.append(float(vertex[3]))
            elif line.startswith('f'):
                face = line.split()
                if len(face) >= 4:
                    # Extract the first index from each group (e.g., 482/482/482)
                    indices = [int(group.split('/')[0]) for group in face[1:]]
                    self.faces.append(indices)

def calculer_axes_inertie(obj_loader):
    # Fonction pour calculer les axes principaux d'inertie
    vertices_array = np.array(obj_loader.vertices)
    if len(vertices_array) == 0:
        raise ValueError("No vertices found in the mesh")

    # Create a temporary file in memory with the content of the OBJ file
    temp_file = StringIO(obj_loader.file_content)
    
    # Load the mesh using trimesh.load
    mesh = trimesh.load(file_obj=temp_file, file_type='obj')

    if mesh.is_empty or mesh.is_volume:
        raise ValueError("Invalid mesh for inertia calculations")

    inertia_tensor = mesh.moment_inertia
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Return the principal axes of inertia as a numpy array
    axes_inertie = eigenvectors.T
    return axes_inertie

def calculer_moments(obj_loader, axe_principal):
    # Fonction pour calculer les moments le long du premier axe principal
    vertices_array = np.array(obj_loader.vertices)
    if len(vertices_array) == 0:
        raise ValueError("No vertices found in the mesh")

    moments = np.sum((vertices_array - np.mean(vertices_array, axis=0)) * axe_principal[0], axis=1)
    return moments

def calculer_distance_moyenne(obj_loader, axe_principal):
    # Fonction pour calculer la distance moyenne des faces au premier axe principal
    vertices_array = np.array(obj_loader.vertices)
    distance_moyenne = np.mean(np.abs(np.dot(vertices_array - np.mean(vertices_array), axe_principal[0])))
    return distance_moyenne

def calculer_variance_distance(obj_loader, axe_principal):
    # Fonction pour calculer la variance de la distance des faces au premier axe principal
    vertices_array = np.array(obj_loader.vertices)
    variance_distance = np.var(np.abs(np.dot(vertices_array - np.mean(vertices_array), axe_principal[0])))
    return variance_distance

def reduire_maillage(obj_loader, target_number_of_faces):
    # Function to reduce the 3D mesh
    vertices_array = np.array(obj_loader.vertices)
    faces_array = np.array(obj_loader.faces) - 1  # Adjusting to 0-indexing

    # Load the mesh
    mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)

    # Check if the mesh can be simplified
    if not mesh.is_watertight or len(mesh.faces) <= target_number_of_faces:
        print("Mesh cannot be simplified further or is already below target face count")
        return obj_loader

    # Simplify the mesh
    try:
        simplified_mesh = mesh.simplify_quadratic_decimation(target_number_of_faces)
    except Exception as e:
        print(f"Error during mesh simplification: {e}")
        return obj_loader

    # Update obj_loader with simplified mesh data
    obj_loader.vertices = simplified_mesh.vertices.tolist()
    obj_loader.faces = (simplified_mesh.faces + 1).tolist()  # Adjusting back to 1-indexing

    return obj_loader

@app.route('/upload', methods=['POST'])
def calcul_file():
    if request.method == 'POST':
        # Retrieve the list of uploaded files
        files = request.files.getlist('file')

        results = []

        for f in files:
            # Ensure the file has a filename
            if f.filename == '':
                continue

            # Construct file paths
            image_file_path = "C:/Users/emy7u/Downloads/3DPotteryDataset_v_1/Thumbnails/All Models/" + f.filename
            obj_file_path = "C:/Users/emy7u/Downloads/3DPotteryDataset_v_1/3D Models/All Models/" + f.filename[:-4] + ".obj"
            obj_file_content = get_file_content(obj_file_path)

            obj_loader = CustomObjLoader(obj_file_content)
            #obj_loader = reduire_maillage(obj_loader)  # Simplify the mesh
            axes_inertie = calculer_axes_inertie(obj_loader)
            moments = calculer_moments(obj_loader, axes_inertie)
            distance_moyenne = calculer_distance_moyenne(obj_loader, axes_inertie)
            variance_distance = calculer_variance_distance(obj_loader, axes_inertie)

            # Store the calculation results in the database
            association = FileAssociation(
                obj_file_path=obj_file_path,
                image_file_path=image_file_path,
                calculation_results=json.dumps({
                    "Axes Principaux d'Inertie": axes_inertie.tolist(),
                    "Moments": moments.tolist(),
                    "Distance Moyenne": distance_moyenne,
                    "Variance de la Distance": variance_distance,
                })
            )

            db.session.add(association)
            db.session.commit()

            result = {
                "id": association.id,
                "message": "Calculation results stored in the database successfully",
                "image_link": url_for('get_image', filename=secure_filename(f.filename), _external=True),
                "obj_link": association.obj_file_path,
                "calculation_results": association.calculation_results
            }

            results.append(result)

        return jsonify(results)
  
def cosine_similarity_between_models1(reference_features, features):
    # Ensure both vectors have the same length
    max_length = max(len(reference_features), len(features))
    reference_features = np.pad(reference_features, (0, max_length - len(reference_features)))
    features = np.pad(features, (0, max_length - len(features)))

    # Calculate cosine similarity
    dot_product = np.dot(reference_features, features)
    norm_reference = np.linalg.norm(reference_features)
    norm_features = np.linalg.norm(features)
    
    similarity = dot_product / (norm_reference * norm_features)
    return similarity

@app.route('/compare', methods=['POST'])
def compare_similarity():
    try:
        if request.method == 'POST':
            start_time = time.time() # Start timing

            # Extract and process reference file
            reference_file_data = request.json['reference']
            print("Reference File Path:", reference_file_data)

            reference_file_content = get_file_content(reference_file_data['obj_link'])
            reference_obj_loader = CustomObjLoader(reference_file_content)

 
            reference_obj_loader = reduire_maillage(reference_obj_loader,target_number_of_faces)  # Simplify the mesh
            reference_axes_inertie = calculer_axes_inertie(reference_obj_loader)
            reference_moments = calculer_moments(reference_obj_loader, reference_axes_inertie)
            reference_distance_moyenne = calculer_distance_moyenne(reference_obj_loader, reference_axes_inertie)
            reference_variance_distance = calculer_variance_distance(reference_obj_loader, reference_axes_inertie)

            # Compare with other models
            comparison_results = []
            for model in request.json['models']:
                model_file_content = get_file_content(model['obj_link'])
                model_obj_loader = CustomObjLoader(model_file_content)
                 
                model_obj_loader = reduire_maillage(model_obj_loader,target_number_of_faces)  # Simplify the mesh
                axes_inertie = calculer_axes_inertie(model_obj_loader)
                moments = calculer_moments(model_obj_loader, axes_inertie)
                distance_moyenne = calculer_distance_moyenne(model_obj_loader, axes_inertie)
                variance_distance = calculer_variance_distance(model_obj_loader, axes_inertie)

                # Calculate cosine similarity with the reference file
                similarity = cosine_similarity_between_models1(
                reference_moments + reference_distance_moyenne + reference_variance_distance,
                moments + distance_moyenne + variance_distance
                )

                image_link = url_for('get_image_by_id', image_id=model['id'], _external=True)
                comparison_results.append({
                    "id": model['id'],
                    "similarity": similarity,
                    "image_link": image_link
                })

            # Sort by similarity
            comparison_results.sort(key=lambda x: x['similarity'], reverse=True)

            end_time = time.time()  # End timing
            total_time = end_time - start_time  # Calculate total time taken
            print(f"Total Time Taken: {total_time} seconds")

            return jsonify(comparison_results)
        
    except Exception as e:
        print(f"Error in comparison: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    app.run(debug=True)