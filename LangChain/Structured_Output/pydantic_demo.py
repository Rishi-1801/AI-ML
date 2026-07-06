from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str='chintu'  # ----> default value
    age: Optional[int]=None  #---> optional
    email: EmailStr   # ----> Built in Validation
    cgpa:float=Field(gt=0,lt=1,default=5)#->restriction

new_student={'name':'Rishi'}

# This throws error
#new_student={'name':20}

# this wont work because email is not valid
#new_student2={'age':12, 'email':'abc'}

# wont work becoz cg is greater than 10
new_student2={'age':12, 'email':'abc@gmail.com','cgpa':12}

new_student2={'age':12, 'email':'abc@gmail.com','cgpa':5}


# new_student2={'age':'12'}  ---> it automatically converts str into int
student=Student(**new_student2)

print(student)

# for converting it into dict or json
#student_dict = dict(student)
#print(student_dict['age'])
#student_json = student.model_dump_json()