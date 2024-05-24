from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse

from carapp.models import PicUpload
from carapp.forms import ImageForm

# Create your views here.
def index(request):

    image_path = ''
    image_path1 = ''
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imagefile=request.FILES['imagefile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('index'))

    else:
        form = ImageForm()

    documents = PicUpload.objects.all()
    for document in documents:
        image_path = document.imagefile.name
        image_path1 = '/'+image_path

        document.delete()

    request.session['image_path'] = image_path

    return render(request, 'index.html',
    {'documents' : documents, 'image_path1': image_path, 'form': form}
    )




                                                            # Detection part Start

import os
import json

import h5py
import numpy as np
import pickle as pk
from PIL import Image

#keras imports
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras import backend as K
import tensorflow as tf


                                                            # prepare image for processing

def prepare_img_224(img_path):
    img = load_img(img_path, target_size = (224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return x


with open('static/cat_counter.pk', 'rb') as f:
    cat_counter = pk.load(f)


cat_list = [k for k, v in cat_counter.most_common()[:27]]

global graph
graph = tf.compat.v1.get_default_graph()


                                                            #Preparing Flat Image

def prepare_flat(img_224):
    base_model = load_model('static/vgg16.h5')
    model = Model(base_model.input, base_model.get_layer('fc1').output)
    feature = model.predict(img_224)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis = 0)
    return flat



                                                            #Loading Models, weights and categories done


CLASS_INDEX_PATH = 'static/imagenet_class_index.json'

def get_predictions(preds, top = 5):

    global CLASS_INDEX
    CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key = lambda x: x[2], reverse = True)
        results.append(result)
    return results


def car_categories_check(img_224):
    first_check = load_model('static/vgg16.h5')
    print("Validating that this is the image of your car...")
    out = first_check.predict(img_224)
    top = get_predictions(out, top = 5)
    for j in top[0]:
        if j[0:2] in cat_list:
            print("Car Check Passed!!!")
            print("\n")
            return True
    return False

                                             # First Check Ends


                                             #Second Check



def car_damage_check(img_flat):
    second_check = pk.load(open('static/SecondCheckClassifier.pickle', 'rb'))
    print("Validating that Damage exists...")
    train_labels = ['00-damage', '01-whole']
    preds = second_check.predict(img_flat)
    predictions = train_labels[preds[0]]

    if train_labels[preds[0]] == '00-damage':
        print("Validation complete -proceeding to locaion and severity determination")
        print("\n")
        return True
    else:
        return False

                                             # Second Check Ends


                                             #Third checklist

def location_assessment(img_flat):
    print("Validating the Damage Area...")
    third_check = pk.load(open('static/ThirdCheckClassifier.pickle', 'rb'))
    train_labels = ['Front', 'Rear', 'Side']
    preds = third_check.predict(img_flat)
    prediction = train_labels[preds[0]]
    print("Your Car is damaged at - " + train_labels[preds[0]])
    print("Location Assessment complete")
    print("\n")
    return prediction


                                        # Third Check Ends


                                        #Fourth Check


def severity_assessment(img_flat):
    print("Validating the Severity...")
    fourth_check = pk.load(open('static/FourthCheckClassifier.pickle', 'rb'))
    train_labels = ['Minor', 'Moderate', 'Severe']
    preds = fourth_check.predict(img_flat)
    prediction = train_labels[preds[0]]
    print("Your Car damage impact is - " + train_labels[preds[0]])
    print("Severity Assessment complete")
    print("\n")
    return prediction


                                            #Fourth Check Ends


                                            #Integration


#load Models

def engine(request):
    MyCar = request.session['image_path']
    img_path = MyCar
    request.session.pop('image_path', None)
    request.session.modified = True
    with graph.as_default():

        img_224 = prepare_img_224(img_path)
        img_flat = prepare_flat(img_224)
        g1 = car_categories_check(img_224)
        g2 = car_damage_check(img_flat)


        while True:
            try:

                if g1 is False:
                    g1_pic = "Image is not of a Car"
                    g2_pic = 'N/A'
                    g3 = 'N/A'
                    g4 = 'N/A'
                    # ns = 'N/A'
                    break
                else:
                    g1_pic = "Its a Car"


                if g2 is False:
                    g2_pic = 'Car is not Damaged'
                    g3 = 'N/A'
                    g4 = 'N/A'
                    # ns = 'N/A'
                    break
                else:
                    g2_pic = "Car is Damaged"

                    g3 = location_assessment(img_flat)
                    g4 = severity_assessment(img_flat)
                    break

            except:
                break

    src = 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".jpg"):
            os.remove(src + image_file_name)

    K.clear_session()

    context = {'g1_pic': g1_pic, 'g2_pic': g2_pic, 'loc': g3, 'sev': g4}

    results = json.dumps(context)
    return HttpResponse(results, content_type = 'application/json')

    # return render(
    # request,
    # 'results.html', context = {'g1_pic':g1_pic, 'g2_pic':g2_pic, 'loc': g3, 'sev': g4}
    # )


                                                #Integration Ends
