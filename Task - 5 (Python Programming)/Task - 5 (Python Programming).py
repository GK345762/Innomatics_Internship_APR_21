#!/usr/bin/env python
# coding: utf-8

# ### 1. Detect Floating Point Number
# 
# 

# In[2]:


import re

class Main():
    def __init__(self):
        self.n = int(input())
        
        for i in range(self.n):
            self.s = input()
            print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', self.s)))
                    
if __name__ == '__main__':
    obj = Main()


# ### 2. Re.split()
# 
# 

# In[4]:


regex_pattern = r"[.,]+"

import re
print("\n".join(re.split(regex_pattern, input("Enter a string s consisting only of digits 0-9, commas ,, and dots .  : "))))


# ### 3. Group(), Groups() & Groupdict()
# 
# 

# In[6]:


print("Print the first occurrence of the repeating character. If there are no repeating characters, print -1. :")

import re

expression=r"([a-zA-Z0-9])\1+"

m = re.search(expression,input("Enter string : "))

if m:
    print(m.group(1))
else:
    print(-1)


# ### 4. Re.findall() & Re.finditer()
# 
# 

# In[8]:


import re
consonants = 'qwrtypsdfghjklzxcvbnm'
vowels = 'aeiou'
match = re.findall(r'(?<=['+consonants+'])(['+vowels+']{2,})(?=['+consonants+'])',input("Enter string : "),flags = re.I)
if match:
    for i in match:
        print(i)
else:
    print -1


# ### 5. Re.start() & Re.end()
# 
# 

# In[9]:


import re
S = input("Enter string S : ")
k = input("Enter string k : ")
anymatch = 'No'
for m in re.finditer(r'(?=('+k+'))',S):
    anymatch = 'Yes'
    print (m.start(1),m.end(1)-1)
if anymatch == 'No':
    print (-1, -1)

    


# ### 6. Regex Substitution
# 
# 

# In[12]:


import re 

for _ in range(int(input())):
    str_ = input()
    str_ = re.sub(r"(?<= )(&&)(?= )", "and", str_)
    print(re.sub(r"(?<= )(\|\|)(?= )", "or", str_))


# ### 7. Validating Roman Numerals
# 
# 

# In[14]:


thousand = 'M{0,3}'
hundred = '(C[MD]|D?C{0,3})'
ten = '(X[CL]|L?X{0,3})'
digit = '(I[VX]|V?I{0,3})'
regex_pattern = r"%s%s%s%s$" % (thousand, hundred, ten, digit)

import re
print(str(bool(re.match(regex_pattern, input("Enter a string of roman characters : ")))))


# ### 8. Validating phone numbers
# 
# 

# In[18]:




import re

N = int(input("Enter number of inputs : "))

for i in range(N):
    number = input("Enter mobile number : ")
    if(len(number)==10 and number.isdigit()):
        output = re.findall(r"^[789]\d{9}$",number)
        if(len(output)==1):
            print("YES")
        else:
            print("NO")
    else:
        print("NO")


# ### 9. Validating and Parsing Email Addresses
# 
# 

# In[19]:


import re
import email.utils
for _ in range(int(input("Enter number of email address : "))):
    s = input("Enter a name and an email address as two space-separated values : ")
    u = email.utils.parseaddr(s)
    if re.search("^[a-z][\w.-]+@[a-z]+\.[a-z]{1,3}$",u[-1],re.I):
        print(s)


# ### 10. Hex Color Code
# 
# 

# In[21]:


import re

T = int(input("Enter the number of code lines : "))
in_css = False
for _ in range(T):
    s = input("Enter the code line by line : ")
    if '{' in s:
        in_css = True
    elif '}' in s:
        in_css = False
    elif in_css:
        for color in re.findall('#[0-9a-fA-F]{3,6}', s):
            print(color)


# ### 11. HTML Parser - Part 1
# 
# 

# In[28]:


from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :',tag)
        for attr in attrs:
                print('->',' > '.join(map(str,attr)))
    def handle_endtag(self, tag):
        print('End   :',tag)
    def handle_startendtag(self, tag, attrs):
        print('Empty :',tag)
        for attr in attrs:
                print('->',' > '.join(map(str,attr)))

html = ""
for i in range(int(input("Enter the number of lines in a HTML code snippet : "))):
    html += input("Enter HTML code lines : ")
                    
                
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# ### 12. HTML Parser - Part 2
# 
# 

# In[29]:


from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):

   
    def handle_comment(self, data):
        if (len(data.split('\n')) != 1):
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data.replace("\r", "\n"))
    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)
    
  
html = ""       
for i in range(int(input("Enter the number of lines in a HTML code snippet : "))):
    html += input("Enter HTML code lines : ").rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# ### 13. Detect HTML Tags, Attributes and Attribute Values
# 
# 

# In[33]:


from html.parser import HTMLParser

class MyHTML(HTMLParser) :
 def handle_starttag(self, tag, attrs):
  print(tag)
  for attr in attrs :
   print("->",attr[0],">",attr[1])
N = int(input("Enter the number of lines in a HTML code snippet : "))
html = ""
parser = MyHTML()
for _ in range(N) :
 line = str(input("Enter HTML code lines : ")).strip()
 parser.feed(line)


# ### 14. Validating UID
# 
# 

# In[34]:


import re

if __name__ == "__main__":
    t = int(input("Enter  the number of test cases : ").strip())
    
    for _ in range(t):
        uid = "".join(sorted(input("Enter employee's UID : ")))
        if (len(uid) == 10 and
            re.match(r'', uid) and 
            re.search(r'[A-Z]{2}', uid) and
            re.search(r'\d\d\d', uid) and
            not re.search(r'[^a-zA-Z0-9]', uid) and
            not re.search(r'(.)\1', uid)):
            print("Valid")
        else:
            print("Invalid")


# ### 15. Validating Credit Card Numbers
# 
# 

# In[35]:


import re
for i in range(int(input("Enter integer N : "))):
    S = input("Enter credit card numbers : ").strip()
    pre_match = re.search(r'^[456]\d{3}(-?)\d{4}\1\d{4}\1\d{4}$',S)
    if pre_match:
        processed_string = "".join(pre_match.group(0).split('-'))
        final_match = re.search(r'(\d)\1{3,}',processed_string)
        if final_match:
            print('Invalid')
        else :
            print('Valid')
    else:
        print('Invalid')


# ### 16. Validating Postal Codes
# 
# 

# In[39]:


regex_integer_in_range = r"^[1-9][\d]{5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"


import re
P = input("Enter postal code : ")

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)


# ### 17. Matrix Script
# 
# 

# In[41]:




import math
import os
import random
import re
import sys




first_multiple_input = input("Enter space-separated integers N and M : ").rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input("Enter the row elements of the matrix script : ")
    matrix.append(matrix_item)

matrix = list(zip(*matrix))

sample = str()
for subset in matrix:
    for letter in subset:
        sample += letter

print(re.sub(r'(?<=\w)([^\w\d]+)(?=\w)', ' ', sample))


# In[ ]:




