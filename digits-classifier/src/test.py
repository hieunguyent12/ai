import pickle


def run():
    Omkar = {"key": "Omkar", "name": "Omkar Pathak", "age": 21, "pay": 40000}
    # database
    db = {}
    db["Omkar"] = Omkar
    output = open("data.pkl", "wb")
    pickle.dump(db, output)
    output.close()


if __name__ == "__main__":
    run()
