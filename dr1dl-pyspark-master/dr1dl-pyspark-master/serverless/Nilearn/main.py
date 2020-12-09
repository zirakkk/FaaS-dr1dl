
from nilearn import decomposition
from nilearn import datasets
from google.cloud import storage
import sklearn
from nilearn import input_data
from nilearn.connectome import ConnectivityMeasure
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import timeit

adhd_data = None

def nilearn(request):
    global adhd_data
    start = timeit.default_timer()
    canica = decomposition.CanICA(n_components=20, mask_strategy="background")
   
    num = 6    
    if adhd_data == None:
        print("None")
        adhd_data = datasets.fetch_adhd(n_subjects=num)
    
    print("1")
    func = adhd_data["func"]
    print("2")
    
    canica.fit(func)
    print("3")
    components = canica.components_
    print("4")
    components_img = canica.masker_.inverse_transform(components)
    print("5")
    masker = input_data.NiftiMapsMasker(components_img, smoothing_fwhm=6,standardize=False, detrend=True, t_r=2.5, low_pass=0.1, high_pass=0.01)
    print("6")
    subjects = []
    adhds = []
    sites = []
    labels = []

    for func_file, confound_file, phenotypic in zip(adhd_data.func, adhd_data.confounds, adhd_data.phenotypic):
        time_series = masker.fit_transform(func_file, confounds=confound_file)
        subjects.append(time_series)
        is_adhd = phenotypic["adhd"]
        if is_adhd == 1:
            adhds.append(time_series)
        sites.append(phenotypic["site"])
        labels.append(phenotypic["adhd"])

    print("7")
    correlation_measure = ConnectivityMeasure(kind="correlation")
    print("8")
    correlation_matrices = correlation_measure.fit_transform(subjects)
    print("9")
    X_train, X_test, y_train, y_test = train_test_split(correlation_matrices, labels, test_size=0.3)

    print("10")
    classifier = keras.models.Sequential()
    print("11")
    classifier.add(keras.layers.Dense(16, activation="relu", kernel_initializer="random_normal"))
    classifier.add(keras.layers.Dense(16, activation="relu", kernel_initializer="random_normal"))
    classifier.add(keras.layers.Dense(1, activation="sigmoid", kernel_initializer="random_normal"))
    print("12")
    classifier.compile(optimizer = keras.optimizers.Adam(lr =.0001),loss="binary_crossentropy", metrics =["accuracy"])
    print("13")

    classifier.fit(np.array(X_train),np.array(y_train), batch_size=1, epochs=10)
    print("14")
    eval_model=classifier.evaluate(np.array(X_train), np.array(y_train))
    print("15")
    print(eval_model)

    y_pred=classifier.predict(X_test,batch_size=32)
    print("16")
    y_pred =(y_pred>0.5)
    print("17")
    print(y_pred)
    print("The End")
          
    stop = timeit.default_timer()
    runtime = stop - start
    print("num: ",num)
    print("Code run time: ",runtime)
    
    return "All good"
