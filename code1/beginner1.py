import tensorflow as tf

mnist = tf.keras.datasets.mnist

# 将样本数据从整数转换为浮点数：
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 通过堆叠层来构建 tf.keras.Sequential (顺序）模型。
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# 对于每个样本，模型都会返回一个包含 logits 或 log-odds 分数的向量，每个类一个。
predictions = model(x_train[:1]).numpy()
print('predictions',predictions)

# tf.nn.softmax 函数将这些 logits 转换为每个类的概率：

# 可以将tf.nn.softmax烘焙到网格最后一层的激活函数中，虽然这可以使模型输出更易解释，但不建议使用这种方式，
# 因为在使用softmax输出时不可能为所有模型提供精确且数值稳定的损失计算。
print('tf.nn.softmax predictions ',tf.nn.softmax(predictions).numpy())

# 使用losses.SparseCategoricalCrossentropy为训练定义损失函数，它会接受logits向量和True索引，
# 并为每个样本返回一个标量损失。
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)