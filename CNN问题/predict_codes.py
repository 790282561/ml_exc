import numpy as np
from matplotlib import image
from tensorflow.keras.models import load_model

# 载入模型
model = load_model('cnn_model.h')


# 构建预测实例
class PredictModel:
    def __init__(self):
        pass

    def get_anwser(self, pic):
        pic_data = image.imread(pic)
        pic_data = np.array(pic_data)
        pic_data = pic_data.reshape(-1, 28, 28, 1)

        predict_pic_data = model.predict(pic_data)
        final_prediction = [result.argmax() for result in predict_pic_data][0]
        a = 0
        for i in predict_pic_data[0]:
            print(a)
            print('Precent:{:30%}'.format(i))
            a += 1
        return final_prediction


def main():
    pic = PredictModel()
    pic_load = pic.get_anwser('xxx.jpg')
    print('图片中数字为：' + pic_load)


# 进行预测
if __name__ == '_main()_':
    main()