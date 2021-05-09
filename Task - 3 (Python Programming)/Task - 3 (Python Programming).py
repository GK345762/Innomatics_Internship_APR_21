#!/usr/bin/env python
# coding: utf-8

# ### 1. Polar Coordinates
# 
# 
Polar coordinates are an alternative way of representing Cartesian coordinates or Complex Numbers.

z = x + yj
# In[1]:


import cmath;

num = complex(input("Enter a single line containing the complex number : "))
z = complex(num)

print(cmath.polar(z)[0])
print(cmath.polar(z)[1])


# ### 2. Find Angle MBC
# 
# 

# In[2]:


import math

AB = int(input())
BC = int(input())

H = math.sqrt(AB**2 + BC**2)
H = H/2.0
adj = BC/2.0

Output = int(round(math.degrees(math.acos(adj/H))))

Output = str(Output)

print(Output+"°")


# ### 3. Triangle Quest 2
# 
# 
# 
# 
You are given a positive integer N.
Your task is to print a palindromic triangle of size N.

For example, a palindromic triangle of size 5 is:

1
121
12321
1234321
123454321

You can't take more than two lines. The first line (a for-statement) is already written for you.
You have to complete the code using exactly one print statement.
# In[3]:


print("N is the size of palindromic triangle : ")
for i in range(1,int(input("Enter an interger N : "))+1):
    print((10**i//9)**2)


# ### 4. Mod Divmod
# 
# 
One of the built-in functions of Python is divmod, which takes two arguments a and b and returns a tuple containing the quotient of a/b first and then the remainder a.

For example:

>>> print divmod(177,10)
(17, 7)

Here, the integer division is 177/10 => 17 and the modulo operator is 177%10 => 7.
# In[7]:


a = int(input("Enter an interger, a : "))
b = int(input("Enter an integer, b : "))
print(a//b)
print(a%b)
print(divmod(a,b))


# ### 5. Power - Mod Power
# 
# 
So far, we have only heard of Python's powers. Now, we will witness them!

Powers or exponents in Python can be calculated using the built-in power function. Call the power function a**b as shown below:

>>> pow(a,b) 
or
>>> a**b

It's also possible to calculate a**b mod m.

>>> pow(a,b,m)  

This is very helpful in computations where you have to print the resultant % mod.
# In[8]:


a = int(input("Enter an interger, a : "))
b = int(input("Enter an interger, b : "))
m = int(input("Enter an interger, m : "))

print(pow(a,b))

print(pow(a,b,m))


# ### 6. Integers Come In All Sizes
# 
# 

# In[9]:


print("Integers a, b, c, and d are given on four separate lines, respectively.")
A = int(input("Enter a : "))
B = int(input("Enter b : "))
C = int(input("Enter c : "))
D = int(input("Enter d : "))

print((A**B)+(C**D))


# ### 7. Triangle Quest
# 
# 
You are given a positive integer N. Print a numerical triangle of height N-1 like the one below:

1
22
333
4444
55555
......

Can you do it using only arithmetic operations, a single for loop and print statement?

Use no more than two lines. The first line (the for statement) is already written for you. You have to complete the print statement.

Note: Using anything related to strings will give a score of 0.
# In[14]:


print("Print a numerical triangle of height N-1 : ")
N = int(input("Enter an integer N : "))
for i in range(1,N): 
    
    print(int(i * 10**i / 9))


# In[ ]:




