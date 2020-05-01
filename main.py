from flask import Flask, render_template, request
app = Flask(__name__)
import pickle



file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close



@app.route('/', methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form

        #errorMessage = ""
        #if myDict['fever'] == '' or myDict['age'] == '' or myDict['pain'] == '':
            #errorMessage = "Please enter all the values"
           # return render_template('index.html', error = errorMessage)
        radius_mean = float(myDict['radius_mean'])
        texture_mean = float(myDict['texture_mean'])
        perimeter_mean = float(myDict['perimeter_mean'])
        area_mean = float(myDict['area_mean'])
        smoothness_mean = float(myDict['smoothness_mean'])
        compactness_mean = float(myDict['compactness_mean'])
        concavity_mean = float(myDict['concavity_mean'])
        concave_points_mean = float(myDict['concave_points_mean'])
        symmetry_mean = float(myDict['symmetry_mean'])
        fractal_dimension_mean = float(myDict['fractal_dimension_mean'])
        radius_se = float(myDict['radius_se'])
        texture_se = float(myDict['texture_se'])
        perimeter_se = float(myDict['perimeter_se'])
        area_se = float(myDict['area_se'])
        smoothness_se = float(myDict['smoothness_se'])
        compactness_se = float(myDict['compactness_se'])
        concavity_se = float(myDict['concavity_se'])
        concave_points_se = float(myDict['concave_points_se'])
        symmetry_se = float(myDict['symmetry_se'])
        fractal_dimension_se = float(myDict['fractal_dimension_se'])
        radius_worst = float(myDict['radius_worst'])
        texture_worst = float(myDict['texture_worst'])
        perimeter_worst = float(myDict['perimeter_worst'])
        area_worst = float(myDict['area_worst'])
        smoothness_worst = float(myDict['smoothness_worst'])
        compactness_worst = float(myDict['compactness_worst'])
        concavity_worst = float(myDict['concavity_worst'])
        concave_points_worst = float(myDict['concave_points_worst'])
        symmetry_worst = float(myDict['symmetry_worst'])
        fractal_dimension_worst = float(myDict['fractal_dimension_worst'])

        inputFeatures = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html', inf = infProb)
    return render_template('index.html')
    # return 'Hello, World!' + str(infProb)


if __name__ == "__main__":
    app.run(debug=True)
    
#run alias python=/usr/local/bin/python3 to set tp python3
#/usr/local/Cellar/jupyterlab/2.1.1/libexec/bin/python3.8p