import hashlib
import json
import os

# Funzione per hashare la password
def hash_password(password: str) -> str:
    """Hash della password usando sha256 di hashlib."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# Funzione per creare un nuovo utente nel file JSON
def create_user(username: str, password: str, user_type: str, db_file: str = None):
    """Aggiungi un nuovo utente al file JSON con la password hashata e il tipo di utente."""
    if db_file is None:
        db_file = os.path.join(os.getcwd(), 'db.json')  

    # Hash della password
    hashed_password = hash_password(password)

    # Crea il dizionario dell'utente con il tipo
    user_data = {
        username: {
            "password": hashed_password,
            "user_type": user_type
        }
    }

    # Carica il file JSON esistente, se esiste, altrimenti crea un nuovo dizionario
    try:
        with open(db_file, 'r') as f:
            user_db = json.load(f)
    except FileNotFoundError:
        user_db = {}

    # Aggiungi o aggiorna l'utente nel database
    user_db.update(user_data)

    # Salva il file JSON
    with open(db_file, 'w') as f:
        json.dump(user_db, f, indent=4)

    print(f"Utente '{username}' creato con successo con tipo '{user_type}'! Il file è stato salvato in: {db_file}")

# Esegui lo script per creare un nuovo utente
if __name__ == "__main__":
    username = input("Inserisci username: ")
    password = input("Inserisci password: ")
    
    # Selezione del tipo di utente
    print("Seleziona il tipo di utente:")
    print("1. Admin")
    print("2. User")
    user_choice = input("Inserisci il numero corrispondente al tipo di utente: ")

    if user_choice == "1":
        user_type = "admin"
    elif user_choice == "2":
        user_type = "user"
    else:
        print("Scelta non valida. Impostato come 'user' di default.")
        user_type = "user"

    create_user(username, password, user_type)
