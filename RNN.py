#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


words = ["I", "love", "deep", "learning"]
word_to_idx = {word: idx for idx, word in enumerate(words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}


# In[3]:


X_data = ["I", "love", "deep"]
Y_data = "learning"


# In[4]:


def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec

vocab_size = len(words)

X = np.array([one_hot(word_to_idx[word], vocab_size) for word in X_data])
Y = word_to_idx[Y_data]


# In[5]:


np.random.seed(42)

hidden_size = 8


Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  
Whh = np.random.randn(hidden_size, hidden_size) * 0.01 
Why = np.random.randn(vocab_size, hidden_size) * 0.01  
# Biases
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


# In[6]:


def rnn_forward(X):
    h_prev = np.zeros((hidden_size, 1))
    hs = {}
    hs[-1] = h_prev


    for t in range(len(X)):
        x_t = X[t].reshape(-1, 1)
        h_t = np.tanh(np.dot(Wxh, x_t) + np.dot(Whh, h_prev) + bh)
        hs[t] = h_t
        h_prev = h_t


    y_pred = np.dot(Why, h_t) + by
    return y_pred, hs


# In[7]:


def softmax(y):
    exp_y = np.exp(y - np.max(y))
    return exp_y / np.sum(exp_y)

def cross_entropy(pred, target_idx):
    return -np.log(pred[target_idx, 0] + 1e-9)


# In[8]:


def rnn_backward(X, hs, y_pred, target_idx):

    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)


    dy = softmax(y_pred)
    dy[target_idx] -= 1

    dWhy += np.dot(dy, hs[len(X)-1].T)
    dby += dy

    
    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(X))):
        dh = np.dot(Why.T, dy) + dh_next
        dh_raw = (1 - hs[t] ** 2) * dh

        dWxh += np.dot(dh_raw, X[t].reshape(1, -1))
        dWhh += np.dot(dh_raw, hs[t-1].T)
        dbh += dh_raw

        dh_next = np.dot(Whh.T, dh_raw)


    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return dWxh, dWhh, dWhy, dbh, dby


# In[9]:


learning_rate = 0.1
n_epochs = 500

for epoch in range(n_epochs):
    y_pred, hs = rnn_forward(X)
    probs = softmax(y_pred)
    loss = cross_entropy(probs, Y)

    dWxh, dWhh, dWhy, dbh, dby = rnn_backward(X, hs, y_pred, Y)

    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby


# In[10]:


if (epoch+1) % 50 == 0:
        pred_idx = np.argmax(probs)
        pred_word = idx_to_word[pred_idx]
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Prediction: {pred_word}")


# In[ ]:




