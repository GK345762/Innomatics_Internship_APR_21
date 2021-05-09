#!/usr/bin/env python
# coding: utf-8

# ### 1. sWAP cASE
# 
# 
You are given a string and your task is to swap cases. In other words, convert all lowercase letters to uppercase letters and vice versa.

For Example:

Www.HackerRank.com → wWW.hACKERrANK.COM
Pythonist 2 → pYTHONIST 2  
# In[3]:


def change(s):
    if str.islower(s):
        return str.upper(s)
    else:
        return str.lower(s)

def swap_case(s):
    return ('').join(map(change,s))

if __name__ == '__main__':
    s = input("Enter a string : ")
    result = swap_case(s)
    print(result)


# ### 2. String Split and Join
# 
# 
In Python, a string can be split on a delimiter.

Example:

>>> a = "this is a string"
>>> a = a.split(" ") # a is converted to a list of strings. 
>>> print a
['this', 'is', 'a', 'string']

Joining a string is simple:

>>> a = "-".join(a)
>>> print a
this-is-a-string 
# In[6]:


def split_and_join(line):
    a = line.split(" ")
    a = "-".join(a)
    return a

if __name__ == '__main__':

    line = input("Enter a string consisting of space separated words : ")
    result = split_and_join(line)
    print(result)


# ### 3. What's Your Name?
# 
# 
You are given the firstname and lastname of a person on two different lines. Your task is to read them and print the following:

Hello firstname lastname! You just delved into python.

Function Description

Complete the print_full_name function in the editor below.

print_full_name has the following parameters:

string first: the first name
string last: the last name

Prints

string: 'Hello firstname lastname! You just delved into python' where firstname and lastname are replaced with first and last.

# In[7]:


def print_full_name(first, last):
    print("Hello"+" "+first_name+" "+last_name+"!"+" "+"You"+" "+
    "just"+" "+"delved"+" "+"into"+" "+"python.")

if __name__ == '__main__':
    first_name = input("Enter first name : ")
    last_name = input("Enter last name : ")
    print_full_name(first_name, last_name)


# ### 4. Mutations
# 
# 
We have seen that lists are mutable (they can be changed), and tuples are immutable (they cannot be changed).

Let's try to understand this with an example.

You are given an immutable string, and you want to make changes to it.

Example

>>> string = "abracadabra"

You can access an index by:

>>> print string[5]
a

What if you would like to assign a value?

>>> string[5] = 'k' 
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment

How would you approach this?

One solution is to convert the string to a list and then change the value.

Example

>>> string = "abracadabra"
>>> l = list(string)
>>> l[5] = 'k'
>>> string = ''.join(l)
>>> print string
abrackdabra
# In[9]:


def mutate_string(string, position, character):
    n = list(string)
    n[position] = character
    string = "".join(n)
    return string
if __name__ == '__main__':
    s = input("Enter a string : ")
    i, c = input("Enter an integer position, the index location and a string character, separated by a space : ").split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# ### 5. Find a string
# 
# 
In this challenge, the user enters a string and a substring. You have to print the number of times that the substring occurs in the given string. String traversal will take place from left to right, not from right to left.

NOTE: String letters are case-sensitive.

Input Format

The first line of input contains the original string. The next line contains the substring.
# In[12]:


def count_substring(string, sub_string):
    count = 0
    for i in range(len(string)-len(sub_string)+1):
        if (string[i:i+len(sub_string)] == sub_string):
            count += 1
    return count

if __name__ == '__main__':
    string = input("Enter original string : ").strip()
    sub_string = input("Enter substring : ").strip()
    
    count = count_substring(string, sub_string)
    print(count)


# ### 6. String Validators
# 
# 
Python has built-in string validation methods for basic data. It can check if a string is composed of alphabetical characters, alphanumeric characters, digits, etc.

str.isalnum()

This method checks if all the characters of a string are alphanumeric (a-z, A-Z and 0-9).

>>> print 'ab123'.isalnum()
True
>>> print 'ab123#'.isalnum()
False

str.isalpha()

This method checks if all the characters of a string are alphabetical (a-z and A-Z).

>>> print 'abcD'.isalpha()
True
>>> print 'abcd1'.isalpha()
False

str.isdigit()

This method checks if all the characters of a string are digits (0-9).

>>> print '1234'.isdigit()
True
>>> print '123edsd'.isdigit()
False

str.islower()

This method checks if all the characters of a string are lowercase characters (a-z).

>>> print 'abcd123#'.islower()
True
>>> print 'Abcd123#'.islower()
False

str.isupper()

This method checks if all the characters of a string are uppercase characters (A-Z).

>>> print 'ABCD123#'.isupper()
True
>>> print 'Abcd123#'.isupper()
False
# In[13]:


if __name__ == '__main__':
    s = input("Enter a string : ")
    print(any(char.isalnum() for char in s))
    print(any(char.isalpha() for char in s))
    print(any(char.isdigit() for char in s))
    print(any(char.islower() for char in s))
    print(any(char.isupper() for char in s))


# ### 7. Text Alignment
# 
# 
In Python, a string of text can be aligned left, right and center.

.ljust(width)

This method returns a left aligned string of length width.

>>> width = 20
>>> print 'HackerRank'.ljust(width,'-')
HackerRank----------  

.center(width)

This method returns a centered string of length width.

>>> width = 20
>>> print 'HackerRank'.center(width,'-')
-----HackerRank-----

.rjust(width)

This method returns a right aligned string of length width.

>>> width = 20
>>> print 'HackerRank'.rjust(width,'-')
----------HackerRank
# In[16]:


#Replace all ______ with rjust, ljust or center. 
thickness = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# ### 8. Text Wrap
# 
# 

# In[18]:


import textwrap

def wrap(string, max_width):
    return textwrap.fill(string,max_width)

if __name__ == '__main__':
    string, max_width = input("Enter string : "), int(input("Enter the maximum width : "))
    result = wrap(string, max_width)
    print(result)


# ### 9. Designer Door Mat
# 
# 

# In[19]:


N, M = map(int, input("Enter space separated values of N and M : ").split())
for i in range(1, N, 2):
    print((i * ".|.").center(M,"-"))
print("WELCOME".center(M, "-"))
for i in range(N-2, -1, -2):
    print((i * ".|.").center(M, "-"))  


# ### 10. String Formatting
# 
# 

# In[21]:



# String Formatting in Python - HackerRank Solution
def print_formatted(number):
    # your code goes here
    # String Formatting in Python - HackerRank Solution START
    for i in range(1,number+1):
        binlen = len(str(bin(number)))
        octf = oct(i).split('o')
        hexf = hex(i).split('x')
        binf = bin(i).split('b')
        print(i , octf[1] , hexf[1].upper() , binf[1] )
    # String Formatting in Python - HackerRank Solution END
    
if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# ### 11. Alphabet Rangoli
# 
# 

# In[22]:


def print_rangoli(size):
    
    width  = size*4-3
    string = ''

    for i in range(1,size+1):
        for j in range(0,i):
            string += chr(96+size-j)
            if len(string) < width :
                string += '-'
        for k in range(i-1,0,-1):    
            string += chr(97+size-k)
            if len(string) < width :
                string += '-'
        print(string.center(width,'-'))
        string = ''

    for i in range(size-1,0,-1):
        string = ''
        for j in range(0,i):
            string += chr(96+size-j)
            if len(string) < width :
                string += '-'
        for k in range(i-1,0,-1):
            string += chr(97+size-k)
            if len(string) < width :
                string += '-'
        print(string.center(width,'-'))
        
    

if __name__ == '__main__':
    n = int(input("Enter size of the rangoli : "))
    print_rangoli(n)


# ### 12. Capitalize!
# 
# 
# 

# In[28]:


def capitalize(string):
    full_name = string.split(' ')
    return ' '.join((word.capitalize() for word in full_name))


if __name__ == '__main__':
    string = input("Enter full name : ")
    capitalized_string = capitalize(string)
    print(capitalized_string)


# ### 13. The Minion Game
# 
# 

# In[29]:


def minion_game(string):
    
    vowel = 'aeiou'.upper()
    strl = len(string)
    kevin = sum(strl-i for i in range(strl) if string[i] in vowel)
    stuart = strl*(strl + 1)/2 - kevin
    if kevin == stuart:
        print ('Draw')
    elif kevin > stuart:
        print ('Kevin %d' % kevin)
    else:
        print ('Stuart %d' % stuart)
        
if __name__ == '__main__':
    s = input("Enter string : ")
    minion_game(s)


# ### 14. Merge the Tools!
# 
# 

# In[30]:


from collections import OrderedDict

def merge_the_tools(string, k):
    strlen = len(string)

    for i in range(0,strlen,k):

        print(''.join(OrderedDict.fromkeys(string[i:i + k])))
         
if __name__ == '__main__':
    string, k = input("Enter string : "), int(input("Enter length of each substring : "))
    merge_the_tools(string, k)
    


# In[ ]:




