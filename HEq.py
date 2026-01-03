import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow import keras as ks
models = ks.models
layers = ks.layers

def f(t, x, y):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([t,x,y])

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([t,x,y])
            model_val = u(tf.concat([t, x, y], axis=1))

        u_t = tape1.gradient(model_val, t)
        u_x = tape1.gradient(model_val, x)
        u_y = tape1.gradient(model_val, y)

    u_xx = tape2.gradient(u_x, x)
    u_yy = tape2.gradient(u_y, y)
    del tape1
    del tape2

    # Heat Eq
    return u_t - (u_xx + u_yy)

def MSE(pcont, u_i, p_i):
    [tcont, xcont, ycont] = np.split(pcont, 3, axis = 1)
    [t_i, x_i, y_i] = np.split(p_i, 3, axis = 1)

    # sample times t_i for MSEf
    # Boundary/init vals for MSEu

    MSEf = tf.reduce_mean(tf.square(f(tcont, xcont, ycont)))
    MSEu = tf.reduce_mean(tf.square(u(tf.concat([t_i, x_i, y_i], axis = 1)) - u_i ))

    return MSEu + MSEf

def TrainingStep(pcont, u_i, p_i):
    with tf.GradientTape() as tape:
        cost = MSE(pcont, u_i, p_i)
    
    grad = tape.gradient(cost, u.trainable_variables)
    optimizer.apply_gradients(zip(grad, u.trainable_variables))
    return cost

def u_bnd_int():
    # max t in training
    t_max = 1

    # initial vals
    t_int = np.zeros(n_int)
    p_int = np.random.uniform(0, 1, size=(n_int, 2))
    y_int = p_int[:, 1]

    u_int = 100 * np.exp(100*y_int)

    # boundary vals
    t_bnd = np.random.uniform(0,20,size = n_bnd)
    x_bnd = np.random.uniform(0,1,size = n_bnd)
    y_bnd = np.concatenate([np.zeros(n_bnd // 2), np.ones(n_bnd // 2)], axis = 0)
    a = np.column_stack([x_bnd, y_bnd])

    #all 
        # (n_int + n_bnd x 1) - array
    u_i = np.concatenate([u_int, np.zeros(n_bnd // 2), np.ones(n_bnd // 2)], axis = 0)

        # (n_int + n_bnd x 3) - array
    p_i = np.concatenate([np.column_stack([t_int, p_int]), np.column_stack([t_bnd, a])], axis = 0)

    # rand vals for f
        # t_cont, x_cont, y_cont
    p_cont = np.column_stack( [np.random.uniform(0, t_max, size = (n_f, 1)), np.random.uniform(0, 1, size = (n_f, 2))] )

    u_i = tf.convert_to_tensor(u_i, dtype=tf.float32)
    p_i = tf.convert_to_tensor(p_i, dtype=tf.float32)
    p_cont = tf.convert_to_tensor(p_cont, dtype=tf.float32)

    return p_cont, u_i, p_i

def update(frame):
    ax.clear()

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_zlim(0, 100.0)

    T = t * np.ones_like(X)

    points = tf.convert_to_tensor(np.hstack([T, X, Y]) , dtype=tf.float32)
    U = u(points).numpy().reshape(n_x, n_y)
    surface = ax.plot_surface(X, Y, U, cmap='viridis')

    t = frame * dt_sim
    return surface,

####### Model

inputs = layers.Input(shape=(3,))
x = inputs
for _ in range(9):
    x = layers.Dense(20, activation='tanh')(x)
outputs = layers.Dense(1)(x)
u = models.Model(inputs=inputs, outputs=outputs)

####### Train

### Training & Test data
    # randomly generate points for u train and test batch
    # try without batches, then implement batch-loop afterwards

    # y=0 > 0, y=1 > 100, x=0,1 > 0

n_bnd = 50
n_int = 100
n_f = 500

print("Generating initial & boundary values.")

p_cont, u_i, p_i = u_bnd_int()

    # mask for train & test
p = 0.7

train_i = np.random.choice([True, False], size=n_bnd + n_int , p=[p, 1-p])
test_i = np.random.choice([True, False], size=n_bnd + n_int , p=[p, 1-p])

train_cont = np.random.choice([True, False], size=n_f, p=[p, 1-p])
test_cont = np.random.choice([True, False], size=n_f, p=[p, 1-p])

# points for f evaluation
p_cont_train = p_cont[train_cont]
p_cont_test = p_cont[test_cont]

# points for IC/BC evaluation
u_i_train = u_i[train_i]
u_i_test = u_i[test_i]
p_i_train = p_i[train_i]
p_i_test = p_i[test_i]

###

epochs = 5
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

print("In training...")
cost = TrainingStep(p_cont_train, u_i_train, p_i_train)
print("Training finished. End Cost:", cost)

# FOR EPOCH-TRAINING
# for epoch in range(epochs):
#     print("Training epoch", epoch)
#     cost = TrainingStep(p_cont_train, u_i_train, p_i_train)
#     # fvals = 0
#     if epoch == epochs-1:
#         print("Training finished. End Cost:", cost)

####### Test

# for epoch in range(epochs):

####### Plotting

# create meshgrid, evaluate at timestep, animate

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

n_x = 50
n_y = n_x.copy()
x = np.linspace(0, 1, n_x)
y = np.linspace(0, 1, n_y)
X, Y = np.meshgrid(x, y)

t = 0
dt_sim = 0.01 # "s"
dt = 20 # ms
n_t = 30 * 1000 / dt

print("Preparing to animate...")

ani = FuncAnimation(fig, update, frames=n_t, interval=dt, blit=False)

plt.show()