### API trained model
1. Using Fastapi for deploying the model

## Reproduce:
This will launch the server on localhost:8001
```python main.py```

Use this command to send post request:
```curl -X 'POST' \
  'http://localhost:8001/predict?input_data=hello' \
  -H 'accept: application/json' \
  -d ''```
