from typing import TypedDict

class Person(TypedDict):

    name:str
    age:int


# new_person : Person={'name':'Rishi','age':20}

new_person : Person={'name':'Rishi','age':'20'} 
# This will still work

print(new_person)