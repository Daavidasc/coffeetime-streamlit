# CoffeeTime Café: predicción de retorno de clientes

Aplicación web en Streamlit que estima si un cliente de CoffeeTime Café regresará durante los próximos 30 días.

## Datos
`clientes_historicos.csv`: 1,500 clientes con edad, visitas de los últimos 90 días, gasto promedio, días desde la última visita, canal de compra, producto favorito, uso del programa de fidelidad, descuentos utilizados, calificación promedio y frecuencia de compras. La variable objetivo es `regresa_30_dias` (1 = regresa, 0 = no regresa).

## Modelo
Random Forest Classifier dentro de un pipeline con codificación one-hot de las variables categóricas. Se guarda en `model.joblib`.

## Cómo usarlo
1. Instalar las librerías: `pip install -r requirements.txt`
2. Entrenar el modelo: `python train_model.py`
3. Iniciar la aplicación: `streamlit run app.py`

La aplicación muestra la clase estimada y la probabilidad de retorno. Es una estimación del modelo, no una garantía.
