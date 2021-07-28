import json
from random import seed
from random import randint

def getUserById(id):
    with open('users.json') as jsonFile:
        users = json.load(jsonFile)
        for user in users:
            if user['id'] == id:
                return user

    return None

def getUserByEmail(email):
    with open('users.json') as jsonFile:
        users = json.load(jsonFile)
        for user in users:
            if user['email'] == email:
                return user

    return None

def storeUser(email):
    existingUser = getUserByEmail(email)

    if(existingUser != None):
        return existingUser

    seed(1)

    userId = 0
    
    while True:
        randId = randint(1, 1000000000)
        if getUserById(randId) == None:
            userId = randId
            break

    with open('users.json', 'r+') as jsonFile:
        users = json.load(jsonFile)
        newUser = {'id': userId, 'email': email}
        users.append(newUser)
        jsonFile.seek(0)
        json.dump(users, jsonFile)
        return newUser