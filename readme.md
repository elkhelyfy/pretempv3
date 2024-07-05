# run the server
    .venv\Scripts\activate
    python app.py
# test the predict model
    curl -X POST -H "Content-Type: application/json" -d '{"input": [your_input_data]}' http://127.0.0.1:5000/predict
    or
    use postman to send post req to your model

