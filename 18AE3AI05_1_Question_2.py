from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from phe import paillier
import numpy as np

# Federated Learning Settings
NUM_CLIENTS = 3

# Load breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X[:100], y[:100], test_size=0.3)

# Generate Paillier key pair
public_key, private_key = paillier.generate_paillier_keypair()

# Define client function to encrypt data and train model on encrypted data
def train_on_encrypted_data(X_encrypted, y_encrypted):
    # Decrypt encrypted data
    X = [private_key.decrypt(x) for x in X_encrypted]
    y = [private_key.decrypt(y_i) for y_i in y_encrypted]

    # Train model on decrypted data
    clf = SVC(kernel='linear')
    clf.fit(X, y)

    # Return unencrypted model weights
    return clf.coef_[0], clf.intercept_[0]

# Split data into clients
X_clients = np.array_split(X_train, NUM_CLIENTS)
y_clients = np.array_split(y_train, NUM_CLIENTS)

# Federated Learning loop
average_weights = None

print('X encryption started ..... ',end = '')
# Encrypt data for each client
client_X_encrypted = [[public_key.encrypt(x) for x in X_i.flatten()] for X_i in X_clients]
print('Completed.')
print('y encryption started ..... ',end = '')
client_y_encrypted = [public_key.encrypt(y_i) for y_i in y_clients]
print('Completed.')

print('Traininig started ..... ',end = '')
# Train model on encrypted data for each client
client_weights = []
client_intercepts = []
for j in range(NUM_CLIENTS):
    weights, intercept = train_on_encrypted_data(client_X_encrypted[j], client_y_encrypted[j])
    client_weights.append(weights)
    client_intercepts.append(intercept)
print('Completed.')

# Aggregate model weights from clients
total_weights = np.sum(client_weights, axis=0)
total_intercept = np.sum(client_intercepts)
average_weights = total_weights / NUM_CLIENTS
average_intercept = total_intercept / NUM_CLIENTS

# Evaluate model on test set
clf = SVC(kernel='linear', coef0=0, C=1)
clf.coef_ = average_weights
clf.intercept_ = average_intercept
X_test_encrypted = [[public_key.encrypt(x) for x in X_i.flatten()] for X_i in X_test]
y_pred_encrypted = clf.predict(X_test_encrypted)
y_pred = [private_key.decrypt(y_i) for y_i in y_pred_encrypted]
accuracy = sum(y_pred == y_test) / len(y_test)
print(accuracy)
