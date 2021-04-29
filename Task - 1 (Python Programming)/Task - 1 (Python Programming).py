#!/usr/bin/env python
# coding: utf-8

# ### 1. Say "Hello, World!" With Python

# In[1]:


string1 = "Hello, World!"
print(string1)


# ### 2. Python If-Else
Given an integer, n, perform the following conditional actions:

If  is odd, print Weird
If  is even and in the inclusive range of 2 to 5, print Not Weird
If  is even and in the inclusive range of 6 to 20, print Weird
If  is even and greater than 20, print Not Weird

Input Format

A single line containing a positive integer, .

Constraints

1 <= n <= 100
# In[2]:


n = int(input('Enter a number between 1 to 100 : '))

if n % 2 != 0:
    print('Weird')
elif 2 <= n <= 5:
    print('Not Weird')
elif 6 <= n <= 20:
    print('Weird')
elif n > 20:
    print('Not Weird')


# ### 3. Arithmetic Operators
The provided code stub reads two integers from STDIN, a and b. Add code to print three lines where:
1. The first line contains the sum of the two numbers.
2. The second line contains the difference of the two numbers (first - second).
3. The third line contains the product of the two numbers.
# In[3]:


a = int(input('Enter first number : '))
b = int(input('Enter second number : '))
print(a + b)
print(a - b)
print(a * b)


# ### 4. Python: Division
The provided code stub reads two integers, a and b, from STDIN.
Add logic to print two lines. The first line should contain the result of integer division, a // b. The second line should contain the result of float division, a / b.

No rounding or formatting is necessary.
# In[4]:


a = int(input('Enter first number : '))
b = int(input('Enter second number : '))
print(a // b)
print(a / b)


# ### 5. Loops
The provided code stub reads and integer, n, from STDIN. For all non-negative integers i<n, print i**2.
Example

The list of non-negative integers that are less than n=3 is [0,1,2]. Print the square of each number on a separate line.
0
1
4
# In[6]:


n = int(input('Enter a number : '))
for i in range(0,n):
    print(i**2)


# ### 6. Write a function
An extra day is added to the calendar almost every four years as February 29, and the day is called a leap day. It corrects the calendar for the fact that our planet takes approximately 365.25 days to orbit the sun. A leap year contains a leap day.

In the Gregorian calendar, three conditions are used to identify leap years:

The year can be evenly divided by 4, is a leap year, unless:
The year can be evenly divided by 100, it is NOT a leap year, unless:
The year is also evenly divisible by 400. Then it is a leap year.
This means that in the Gregorian calendar, the years 2000 and 2400 are leap years, while 1800, 1900, 2100, 2200, 2300 and 2500 are NOT leap years. Source

Task

Given a year, determine whether it is a leap year. If it is a leap year, return the Boolean True, otherwise return False.

Note that the code stub provided reads from STDIN and passes arguments to the is_leap function. It is only necessary to complete the is_leap function.
# In[7]:


def is_leap(year):
    leap = False
    
    if (year % 400 == 0):
        return True
    if (year % 100 == 0):
        return leap
    if (year % 4 == 0):
        return True
    else:
        return False  
    
    return leap
year = int(input())
print(is_leap(year))


# ### 7. Print Function
The included code stub will read an integer, , from STDIN.

Without using any string methods, try to print the following:

123...n

Note that "..." represents the consecutive values in between.

Example

n = 5

Print the string 12345.
# In[8]:


n = int(input())
for i in range(0,n):
    print (i+1, end="")


# In[ ]:




