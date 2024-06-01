from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import tensorflow_datasets as tfds
import tensorflow as tf
import requests
import json
from datetime import datetime
import numpy as np
import pytz
import os

app = Flask(__name__)
CORS(app)

# 상대 경로로 변경
file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
train_data = pd.read_csv(file_path)

questions = []
for sentence in train_data['Q']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2
MAX_LENGTH = 40

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = float(d_model)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step):
        step_float = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step_float)
        arg2 = step_float * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    if mask is not None:
        logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(CustomMultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        outputs = self.dense(concat_attention)
        return outputs

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

tf.keras.utils.get_custom_objects().update({'CustomSchedule': CustomSchedule})
tf.keras.utils.get_custom_objects().update({'loss_function': loss_function})
tf.keras.utils.get_custom_objects().update({'create_padding_mask': create_padding_mask})
tf.keras.utils.get_custom_objects().update({'create_look_ahead_mask': create_look_ahead_mask})
tf.keras.utils.get_custom_objects().update({'CustomMultiHeadAttention': CustomMultiHeadAttention})
tf.keras.utils.get_custom_objects().update({'PositionalEncoding': PositionalEncoding})

# 상대 경로로 변경
saved_model_path = os.path.join(os.path.dirname(__file__), 'transformer_v7')
loaded_model = tf.saved_model.load(saved_model_path)

def evaluate(sentence, model):
    sentence = preprocess_sentence(sentence)
    sentence = START_TOKEN + tokenizer.encode(sentence) + END_TOKEN
    sentence = tf.cast(sentence, dtype=tf.float32)
    sentence = tf.expand_dims(sentence, axis=0)
    output = tf.expand_dims(START_TOKEN + [tokenizer.vocab_size], axis=0)
    output = tf.cast(output, dtype=tf.float32)
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.equal(predicted_id, END_TOKEN[0]):
            break
        output = tf.concat([output, tf.cast(predicted_id, dtype=tf.float32)], axis=-1)
    return tf.squeeze(output, axis=0)

def predict(sentence, model):
    prediction = evaluate(sentence, model)
    prediction = [int(i) for i in prediction.numpy() if i < tokenizer.vocab_size]
    predicted_sentence = tokenizer.decode(prediction)
    return predicted_sentence

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def fetch_weather_data():
    city = "Seoul"
    apikey = "327389fd33def65a6deed3ebfc2df470"
    api = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}"
    loc = requests.get(api)
    loc = json.loads(loc.text)
    lat = loc['coord']['lat']
    lon = loc['coord']['lon']
    api = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={apikey}"
    result = requests.get(api)
    result = json.loads(result.text)
    return result

def senetence_completion(input, temp):
    timezone = pytz.timezone('Asia/Seoul')
    now = datetime.now(timezone)
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    date_list_after = ['내일', '모레', '글피', '내일 모레', '1일 뒤', '2일 뒤', '3일 뒤', '4일 뒤', '5일 뒤']
    time_list = ['아침', '점심', '저녁', '오전', '오후', '낮', '밤', '새벽']
    hour_list = ['0시', '1시', '2시', '3시', '4시', '5시', '6시', '7시', '8시', '9시', '10시', '11시', '12시',
                 '13시', '14시', '15시', '16시', '17시', '18시', '19시', '20시', '21시', '22시', '23시', '24시']
    temp_need = []
    found_keyword = None
    for keyword in date_list_after:
        if keyword in input:
            found_keyword = keyword
            break
    if found_keyword:
        if found_keyword in ['내일', '1일 뒤']:
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+1 == int(temp['list'][i]['dt_txt'][8]+temp['list'][i]['dt_txt'][9]):
                    temp_need.append(temp['list'][i])
        if found_keyword in ['모레', '2일 뒤', '내일 모레']:
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+2 == int(temp['list'][i]['dt_txt'][8]+temp['list'][i]['dt_txt'][9]):
                    temp_need.append(temp['list'][i])
        if found_keyword in ['글피', '3일 뒤']:
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+3 == int(temp['list'][i]['dt_txt'][8]+temp['list'][i]['dt_txt'][9]):
                    temp_need.append(temp['list'][i])
        if found_keyword in ['4일 뒤']:
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+4 == int(temp['list'][i]['dt_txt'][8]+temp['list'][i]['dt_txt'][9]):
                    temp_need.append(temp['list'][i])
        if found_keyword in ['5일 뒤']:
            for i in range(40):
                if int(formatted_now[8]+formatted_now[9])+5 == int(temp['list'][i]['dt_txt'][8]+temp['list'][i]['dt_txt'][9]):
                    temp_need.append(temp['list'][i])
    else:
        for i in range(40):
            if int(formatted_now[8]+formatted_now[9]) == int(temp['list'][i]['dt_txt'][8]+temp['list'][i]['dt_txt'][9]):
                temp_need.append(temp['list'][i])
    near_time = []
    if (any(hour in input for hour in hour_list)):
        if ("오후" in input) or ("낮" in input) or ("점심" in input) or ("저녁" in input) or ("밤" in input):
            for i in range(len(temp_need)):
                matchs = re.search(r'(\d+)시', input)
                hour_str = matchs.group(1)
                hour = int(hour_str)
                if hour < 12:
                    hour += 12
                near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                near_time.append(abs(hour-near))
            ine = np.argmin(near_time)
        else:
            for i in range(len(temp_need)):
                matchs = re.search(r'(\d+)시', input)
                hour_str = matchs.group(1)
                hour = int(hour_str)
                near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                near_time.append(abs(hour-near))
            ine = np.argmin(near_time)
    else:
        if ("오전" in input) or ("아침" in input):
            for i in range(len(temp_need)):
                near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                near_time.append((near))
            if 6 in near_time:
                ine = near_time.index(6)
            elif 9 in near_time:
                ine = near_time.index(9)
            elif 12 in near_time:
                ine = near_time.index(12)
        elif ("낮" in input) or ("오후" in input) or ("점심" in input):
            for i in range(len(temp_need)):
                near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                near_time.append((near))
            if 15 in near_time:
                ine = near_time.index(15)
            elif 12 in near_time:
                ine = near_time.index(12)
        elif ("저녁" in input) or ("밤" in input):
            for i in range(len(temp_need)):
                near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                near_time.append((near))
            if 18 in near_time:
                ine = near_time.index(18)
            elif 21 in near_time:
                ine = near_time.index(21)
            elif 0 in near_time:
                ine = near_time.index(0)
        elif ("새벽" in input):
            for i in range(len(temp_need)):
                near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                near_time.append((near))
            if 3 in near_time:
                ine = near_time.index(3)
            elif 6 in near_time:
                ine = near_time.index(6)
            elif 0 in near_time:
                ine = near_time.index(0)
        else:
            for i in range(len(temp_need)):
                now = int(formatted_now[11]+formatted_now[12])
                near = int(temp_need[i]['dt_txt'][11]+temp_need[i]['dt_txt'][12])
                near_time.append(abs(now-near))
            ine = np.argmin(near_time)
    i = ine
    timestamp = temp_need[i]['dt_txt']
    temperature = temp_need[i]['main']['temp']
    lowest_temperature = temp_need[i]['main']['temp_min']
    highest_temperature = temp_need[i]['main']['temp_max']
    humidity = temp_need[i]['main']['humidity']
    weather = temp_need[i]['weather'][0]['main']
    rain = temp_need[i]['rain']['3h'] if 'rain' in temp_need[i] else 0
    rain_pop = str(float(temp_need[i]['pop']) * 100)
    visibility = temp_need[i]['visibility']
    wind = temp_need[i]['wind']['speed']
    pressure = temp_need[i]['main']['pressure']
    snow = temp_need[i]['snow']['3h'] if 'snow' in temp_need[i] else 0
    return {
        "timestamp": timestamp,
        "temperature": temperature,
        "lowest_temperature": lowest_temperature,
        "highest_temperature": highest_temperature,
        "humidity": humidity,
        "weather": weather,
        "rain": rain,
        "rain_pop": rain_pop,
        "visibility": visibility,
        "wind": wind,
        "pressure": pressure,
        "snow": snow
    }

def replace_time_in_output(input_str, output_str, time_list):
    for time_word in time_list:
        if time_word in input_str:
            for out_time_word in time_list:
                if out_time_word in output_str:
                    return output_str.replace(out_time_word, time_word)
    return output_str

def remove_space_before_punctuation(text):
    corrected_text = re.sub(r'(?<=\S) (?=[.,!?])', '', text)
    corrected_text = re.sub(r'([.,!?])(\s)(\s*)', r'\1\3', corrected_text)
    return corrected_text

@app.route('/greet', methods=['POST'])
def greet():
    data = request.get_json()
    user_input = data['input']
    temp = fetch_weather_data()
    output1 = predict(user_input, loaded_model)
    output1 = remove_space_before_punctuation(output1)
    output1 = replace_time_in_output(user_input, output1, ['{아침}', '{점심}', '{저녁}', '{오전}', '{오후}', '{낮}', '{밤}', '{새벽}'])
    if "{" in output1:
        output2 = output1
        weather_data = senetence_completion(output1, temp)
        if "{온도}" in output1:
            output2 = output2.replace("{온도}", str(weather_data["temperature"]) + "℃")
        if "{최저기온}" in output1:
            output2 = output2.replace("{최저기온}", str(weather_data["lowest_temperature"]) + "℃")
        if "{최고온도}" in output1:
            output2 = output2.replace("{최고온도}", str(weather_data["highest_temperature"]) + "℃")
        if "{습도}" in output1:
            output2 = output2.replace("{습도}", str(weather_data["humidity"]) + "%")
        if "{날씨}" in output1:
            if weather_data["weather"] == 'Clear': weather_data["weather"] = '맑은 날씨'
            elif weather_data["weather"] == 'Rain': weather_data["weather"] = '비오는 날씨'
            elif weather_data["weather"] == 'Clouds': weather_data["weather"] = '흐린 날씨'
            elif weather_data["weather"] == 'Snow': weather_data["weather"] = '눈오는 날씨'
            output2 = output2.replace("{날씨}", weather_data["weather"])
        if "{강수량}" in output1:
            output2 = output2.replace("{강수량}", str(weather_data["rain"]) + "mm")
        if "{강수확률}" in output1:
            output2 = output2.replace("{강수확률}", str(weather_data["rain_pop"]) + "%")
        if "{가시거리}" in output1:
            output2 = output2.replace("{가시거리}", str(weather_data["visibility"]) + "m")
        if "{풍속}" in output1:
            output2 = output2.replace("{풍속}", str(weather_data["wind"]) + "m/s")
        if "{대기압}" in output1:
            output2 = output2.replace("{대기압}", str(weather_data["pressure"]) + "hPa")
        if "{적설량}" in output1:
            output2 = output2.replace("{적설량}", str(weather_data["snow"]) + "mm")
        output2 = output2.replace("{", "").replace("}", "")
        response = f"예측 시각: {weather_data['timestamp']}\n챗봇: {output2}"
    else:
        response = f"챗봇: {output1.replace('{', '').replace('}', '')}"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
