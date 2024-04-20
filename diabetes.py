import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import googleapiclient.discovery
import os

from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

###### [1] 필요한 라이브러리를 임포트한다.


np.random.seed(42)

###### [2] 랜덤시드를 설정한다.


app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

###### [3] Flask 애플리케이션을 생성한다.


Bootstrap5(app)

###### [4] 부트스트랩의 기능과 스타일을 사용하기 위해 초기화한다.


class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

###### [5] 폼 클래스를 정의한다. 각각의 입력필드와 제출 버튼 필드를 보여준다.


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

###### [6] 루트경로에 대한 함수다. index.html 파일을 렌더링한다.


@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = np.array([[float(form.preg.data),
                            float(form.glucose.data),
                            float(form.blood.data),
                            float(form.skin.data),
                            float(form.insulin.data),
                            float(form.bmi.data),
                            float(form.dpf.data),
                            float(form.age.data)]])
        print(X_test.shape)
        print(X_test)

        data = pd.read_csv('./diabetes.csv', sep=',')

        X = data.values[:, 0:8]
        y = data.values[:, 8]

        scaler = MinMaxScaler()
        scaler.fit(X)

        X_test = scaler.transform(X_test)

        project_id = "ai-project-12-1"
        model_id = "my_pima_model"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ai-project-12-1-94499ca3f190.json"
        model_path = "projects/{}/models/{}".format(project_id, model_id)
        model_path += "/versions/v0001/"
        ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

        input_data_json = {"signature_name": "serving_default", "instances": X_test.tolist()}

        request = ml_resource.predict(name=model_path, body=input_data_json)
        response = request.execute()
        print("\nresponse: \n", response)


        if "error" in response:
            raise RuntimeError(response["error"])

        predD = np.array([pred['dense_2'] for pred in response["predictions"]])
        
        print(predD[0][0])
        res = predD[0][0]
        res = np.round(res, 2)
        res = (float)(np.round(res * 100))

        return render_template('result.html', res=res)

    return render_template('prediction.html', form=form)

###### [7] /prediction 경로에 대한 함수다. 입력필드에 각각의 값을 입력하고 입력한 값들을 출력한다.
###### diabetes 데이터셋을 가져와 학습한다. 앞에 12장에서 만든 pima 모델을 이용하여 예측을 수행한다.
###### 계산 결과값을 response에 저장하고 도출된 값에 100을 곱하여 퍼센테이지로 출력한다.


if __name__ == '__main__':
    app.run()

###### [8] 해당 코드는 파이썬스크립트가 실행될 때만 Flask 애플리케이션을 실행하도록 한다.