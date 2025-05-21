# Monografia Ninapro DB1

## Lista de notebooks

A continuación se lista:
1. **Combinación del dataset**: [00_combinacion_dataset.ipynb](00_combinacion_dataset.ipynb)
2. **Selección de posturas**: [01_seleccion_posturas_dataset.ipynb](01_seleccion_posturas_dataset.ipynb)
3. **Analisis estadistico de las señales**: 
   * **EDA señales crudas**: [02_posturas_dataset_EDA.ipynb](02_posturas_dataset_EDA.ipynb)
   * **EDA señales filtradas**: [02_filtered_posturas_dataset_EDA.ipynb](02_filtered_posturas_dataset_EDA.ipynb)
4. **Extracción de caracteristias**: [03_features_extraction.ipynb](03_features_extraction.ipynb)
5. **Analisis estadistico de las señales extraidas**: [04_features_EDA.ipynb](04_features_EDA.ipynb)


## Para despues

| **Grupo Funcional**                     | **Números de Movimientos** |   |
| --------------------------------------- | -------------------------- | - |
| **1. Movimientos bilaterales**          | 1, 2                       |   |
| **2. Interacción con objetos**          | 3, 4, 5, 6                 |   |
| **3. Uso de utensilios**                | 7, 8, 9, 10                |   |
| **4. Higiene personal**                 | 11, 12, 13, 14             |   |
| **5. Actividades cotidianas simples**   | 15, 16, 17, 18             |   |
| **6. Movimientos funcionales diversos** | 19, 20, 21, 22, 23         |   |

* https://pmc.ncbi.nlm.nih.gov/articles/PMC7039218/

En el articulo anterior, se dedujo la siguiente CNN:

![img](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/27cb/7039218/9dc477431943/sensors-20-00672-g002.jpg)


```py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Conv1D, BatchNormalization, PReLU, Dropout,
    Dense, GlobalAveragePooling1D
)

def build_model_from_diagram(input_shape, num_classes, lstm_units_1=128, lstm_units_2=128, fc_units=64):
    """
    Construye un modelo de Keras basado en el diagrama de arquitectura proporcionado.

    Args:
        input_shape (tuple): Forma de los datos de entrada (sequence_length, num_features).
                             Ej: (52, 10) si la secuencia tiene 52 pasos y 10 características por paso.
        num_classes (int): Número de clases para la capa de salida Softmax.
        lstm_units_1 (int): Número de unidades en la primera capa LSTM.
        lstm_units_2 (int): Número de unidades en la segunda capa LSTM.
        fc_units (int): Número de unidades en la capa densa ("Fully Connected") antes de la PReLU.

    Returns:
        tensorflow.keras.models.Model: El modelo de Keras compilado.
    """

    # Capa de Entrada
    # El diagrama sugiere una longitud de secuencia de 52 (indicado por 1, 2, ..., 52 bajo las LSTMs).
    # num_features debe ser especificado según los datos de entrada.
    inputs = Input(shape=input_shape, name="input_signals")

    # Capas LSTM
    # Dos capas LSTM apiladas. Ambas retornan secuencias completas.
    x = LSTM(units=lstm_units_1, return_sequences=True, name="lstm_1")(inputs)
    x = LSTM(units=lstm_units_2, return_sequences=True, name="lstm_2")(x)
    # Salida de LSTM: (batch_size, sequence_length, lstm_units_2)

    # Bloque 1DCNN (según el desglose a la derecha del diagrama)
    # Este bloque procesa la salida de las capas LSTM.

    # Primera sub-capa convolucional dentro del bloque 1DCNN
    # Conv1D 64 filtros, kernel 3, stride 1, pad 1 (padding='same' lo logra)
    cnn_output = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name="conv1d_1")(x)
    cnn_output = BatchNormalization(name="batchnorm_1")(cnn_output)
    cnn_output = PReLU(name="prelu_1")(cnn_output)

    # Capa Dropout
    cnn_output = Dropout(0.3, name="dropout_1")(cnn_output)

    # Segunda sub-capa convolucional dentro del bloque 1DCNN
    # Conv1D 32 filtros, kernel 3, stride 1, pad 1
    cnn_output = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', name="conv1d_2")(cnn_output)
    cnn_output = BatchNormalization(name="batchnorm_2")(cnn_output)
    cnn_output = PReLU(name="prelu_2")(cnn_output)
    # Salida del bloque 1DCNN: (batch_size, sequence_length, 32)

    # Para pasar a una capa "Fully Connected" estándar para clasificación de secuencia,
    # usualmente se reduce la dimensión temporal. GlobalAveragePooling1D es una opción común.
    pooled_output = GlobalAveragePooling1D(name="global_avg_pooling")(cnn_output)
    # Salida de Pooling: (batch_size, 32)

    # Capa "Fully Connected"
    # El diagrama muestra "Fully Connected", luego una flecha a "Softmax".
    # Asumimos que "Fully Connected" incluye una activación PReLU como en otras partes.
    fc_output = Dense(units=fc_units, name="fully_connected_dense")(pooled_output)
    fc_output = PReLU(name="prelu_fc")(fc_output)
    # Salida de FC: (batch_size, fc_units)

    # Capa de Salida Softmax
    outputs = Dense(units=num_classes, activation='softmax', name="output_softmax")(fc_output)
    # Salida final: (batch_size, num_classes)

    # Crear el modelo
    model = Model(inputs=inputs, outputs=outputs, name="SequentialSignalClassifier")

    return model

if __name__ == '__main__':
    # Ejemplo de cómo usar la función para construir el modelo

    # Parámetros de ejemplo (debes ajustarlos a tu problema específico)
    sequence_length = 52  # Según el diagrama
    num_input_features = 10 # Ejemplo: si tus señales de entrada tienen 10 características por paso de tiempo
    num_classes = 5         # Ejemplo: si tienes 5 clases de salida
    
    # Dimensiones de las capas (puedes experimentar con estos valores)
    lstm_units_1 = 64
    lstm_units_2 = 64
    fc_layer_units = 32

    # Construir el modelo
    example_model = build_model_from_diagram(
        input_shape=(sequence_length, num_input_features),
        num_classes=num_classes,
        lstm_units_1=lstm_units_1,
        lstm_units_2=lstm_units_2,
        fc_units=fc_layer_units
    )

    # Imprimir un resumen del modelo
    example_model.summary()

    # Para entrenar este modelo, necesitarías compilarlo primero:
    # example_model.compile(optimizer='adam', 
    #                       loss='categorical_crossentropy', # o 'sparse_categorical_crossentropy' si tus etiquetas son enteros
    #                       metrics=['accuracy'])
    
    # Y luego entrenarlo con tus datos:
    # history = example_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
```

En ChatGPT

Input signals (52 canales) 
→ LSTM paralelas (por canal) 
→ Concatenación o integración de salidas 
→ CNN 1D (Conv1D + PReLU + BatchNorm + Dropout) 
→ Fully Connected 
→ Softmax (Clasificación)

```py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_CNN_Model(nn.Module):
    def __init__(self, input_channels=52, sequence_length=100, num_classes=10):
        super(LSTM_CNN_Model, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.hidden_size = 64  # LSTM hidden size
        self.lstm_layers = 1

        # LSTM: procesamos cada canal por separado
        self.lstm = nn.LSTM(
            input_size=sequence_length,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True
        )

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.prelu1 = nn.PReLU()

        self.dropout = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.prelu2 = nn.PReLU()

        # Fully connected
        self.fc = nn.Linear(64 * self.hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, channels=52, sequence_length)
        batch_size = x.size(0)

        # Paso 1: LSTM → convertimos canales en batch
        x = x.view(batch_size * self.input_channels, self.sequence_length).unsqueeze(-1)  # (batch*52, seq_len, 1)
        x, _ = self.lstm(x)  # Output shape: (batch*52, seq_len, hidden)
        x = x[:, -1, :]  # Tomamos solo la última salida temporal
        x = x.view(batch_size, self.input_channels, self.hidden_size)  # (batch, 52, hidden)

        # Paso 2: CNN
        x = x.permute(0, 2, 1)  # (batch, hidden, 52)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)

        # Paso 3: Fully connected
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        # Paso 4: Softmax
        return F.log_softmax(x, dim=1)


model = LSTM_CNN_Model(input_channels=52, sequence_length=100, num_classes=10)
dummy_input = torch.randn(8, 52, 100)  # batch_size=8
output = model(dummy_input)
print(output.shape)  # Debe ser: (8, 10)
```

Tengo lo siguiente:

```py
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Conv1D, BatchNormalization, PReLU, Dropout, Dense, Flatten
from tensorflow.keras.models import Model

def build_lstm_cnn_model(sequence_length=100, input_channels=52, num_classes=10):
    # Entrada: (batch_size, sequence_length, input_channels)
    inputs = Input(shape=(sequence_length, input_channels))

    # Paso 1: LSTM bidireccional por canal (canales = features)
    # Transponer para aplicar LSTM por canal: (batch, features=52, sequence)
    x = tf.transpose(inputs, perm=[0, 2, 1])

    # Aplicar LSTM por canal
    lstm_out = []
    for i in range(input_channels):
        single_channel = tf.keras.layers.Lambda(lambda z: z[:, i, :])(x)  # (batch, sequence_length)
        single_channel = tf.expand_dims(single_channel, -1)
        lstm = LSTM(units=64, return_sequences=False)(single_channel)
        lstm_out.append(lstm)

    # Concatenar salidas de los 52 LSTM
    x = tf.keras.layers.Concatenate()(lstm_out)  # (batch_size, 52 * 64)

    # Reorganizar a forma compatible con Conv1D: (batch, steps=52, channels=64)
    x = tf.keras.layers.Reshape((input_channels, 64))(x)

    # Paso 2: 1D CNN
    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Dropout(0.3)(x)

    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    # Paso 3: Fully Connected
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=x)

model = build_lstm_cnn_model(sequence_length=100, input_channels=52, num_classes=10)
model.summary()


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Dummy data para pruebas
import numpy as np
X_dummy = np.random.rand(8, 100, 52).astype(np.float32)
y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, 10, 8), num_classes=10)

model.fit(X_dummy, y_dummy, epochs=1)


```


* https://prsdm.github.io/mlops-project/
