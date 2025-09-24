from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from io import BytesIO
import os

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

    elif request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # 画像の読み込みと前処理
                img_file = form.cleaned_data['image']
                img_file = BytesIO(img_file.read())
                img = load_img(img_file, target_size=(224, 224))  # VGG16仕様
                img_array = img_to_array(img).reshape((1, 224, 224, 3))
                img_array = preprocess_input(img_array)

                # モデルの読み込み
                model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
                model = load_model(model_path)

                # 予測と上位5カテゴリの取得
                result = model.predict(img_array)
                top_predictions = decode_predictions(result, top=5)[0]

                # JavaScriptから送られた画像データ（Base64）を取得
                img_data = request.POST.get('img_data')

                return render(request, 'home.html', {
                    'form': form,
                    'top_predictions': top_predictions,
                    'img_data': img_data
                })

            except Exception as e:
                form.add_error(None, f'予測処理中にエラーが発生しました: {str(e)}')
                return render(request, 'home.html', {'form': form})

        else:
            return render(request, 'home.html', {'form': ImageUploadForm()})

    else:
        return render(request, 'home.html', {'form': ImageUploadForm()})