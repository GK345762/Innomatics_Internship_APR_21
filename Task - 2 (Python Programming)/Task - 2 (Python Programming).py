#!/usr/bin/env python
# coding: utf-8

# ### 1. List Comprehensions
Let's learn about list comprehensions! You are given three integers x, y and z representing the dimensions of a cuboid along with an integer n. Print a list of all possible coordinates given by (i,j,k) on a 3D grid where the sum of i+j+k is not equal to n. Here, 0 <= i <=x; 0 <= j <= y; 0 <= k <= z. Please use list comprehensions rather than multiple loops, as a learning exercise.

Example

x = 1
y = 1
z = 2
n = 3

All permutations of [i,j,k] are : 

[[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[0,1,2],[1,0,0],[1,0,1],[1,0,2],[1,1,0],[1,1,1],[1,1,2]].

Print an array of the elements that do not sum to n =3

[[0,0,0],[0,0,1],[0,0,2],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,2]
# In[1]:


print('Three integers x, y and z represents the dimensions of a cuboid along with an integer n')
x = int(input('x : '))
y = int(input('y : '))
z = int(input('z : '))
n = int(input('n : '))
print( [[i,j,k] for i in range( x + 1) for j in range( y + 1) for k in range(z+1) if ( ( i + j + k ) != n ) ])


# ### 2. Find the Runner-Up Score!
Given the participants' score sheet for your University Sports Day, you are required to find the runner-up score. You are given n scores. Store them in a list and find the score of the runner-up.

Input Format

The first line contains n. The second line contains an array A[] of n integers each separated by a space.

Constraints

2 <= n <= 10
-100 <= A[i] <= 100

Output Format

Print the runner-up score.

Sample Input 0

5
2 3 6 6 5

Sample Output 0

5

Explanation 0

Given list is [2,3,6,6,5]. The maximum score is 6, second maximum is 5. Hence, we print 5 as the runner-up score.
# In[2]:


n = int(input("Enter Number of participants : "))

arr = map(int, input('Enter n number of scores one after another separated by spaces :').split())    
print(sorted(list(set(arr)))[-2])


# ### 3. Nested Lists
# 
# 
Given the names and grades for each student in a class of N students, store them in a nested list and print the name(s) of any student(s) having the second lowest grade.

Note: If there are multiple students with the second lowest grade, order their names alphabetically and print each name on a new line.

Example

records = [["chi",20.0],["beta",50.0],["alpha",50.0]]

The ordered list of scores is [20.0,50.0], so the second lowest score is 50.0. There are two students with that score: ["beta","alpha"]. Ordered alphabetically, the names are printed as:

alpha
beta

Input Format

The first line contains an integer, N, the number of students.
The 2N subsequent lines describe each student over 2 lines.
- The first line contains a student's name.
- The second line contains their grade.

Constraints

2 <= N <= 5
There will always be one or more students having the second lowest grade.

Output Format

Print the name(s) of any student(s) having the second lowest grade in. If there are multiple students, order their names alphabetically and print each one on a new line.

Sample Input 0

5
Harry
37.21
Berry
37.21
Tina
37.2
Akriti
41
Harsh
39

Sample Output 0

Berry
Harry

Explanation 0

There are 5 students in this class whose names and grades are assembled to build the following list:

python students = [['Harry', 37.21], ['Berry', 37.21], ['Tina', 37.2], ['Akriti', 41], ['Harsh', 39]]

The lowest grade of 37.2 belongs to Tina. The second lowest grade of 37.21 belongs to both Harry and Berry, so we order their names alphabetically and print each name on a new line.
# In[9]:


score_list = [];
for score in range(int(input('Enter number of students : '))):
    name = input('Enter Name : ')
    score = float(input('Enter Score : '))
        
    score_list.append([name, score])
second_highest = sorted(set([score for name, score in score_list]))[1]
print('\n'.join(sorted([name for name, score in score_list if score == second_highest])))


# ### 4. Finding the percentage
The provided code stub will read in a dictionary containing key/value pairs of name:[marks] for a list of students. Print the average of the marks array for the student name provided, showing 2 places after the decimal.
# In[17]:


n = int(input('Enter number of students : '))
student_marks = {}
for i in range(n):
    line = input().split()
    name, scores = line[0], line[1:]
    scores = map(float, scores)
    student_marks[name] = scores
    
query_name = input()
total_marks = sum(student_marks[query_name])
average_marks = total_marks/3 
print("%.2f"%(average_marks))


# ### 5. Lists
Consider a list (list = []). You can perform the following commands:

1. insert i e: Insert integer e at position i.
2. print: Print the list.
3. remove e: Delete the first occurrence of integer e.
4. append e: Insert integer e at the end of the list.
5. sort: Sort the list.
6. pop: Pop the last element from the list.
7. reverse: Reverse the list.

Initialize your list and read in the value of n followed by n lines of commands where each command will be of the 7 types listed above. Iterate through each command in order and perform the corresponding operation on your list.
# In[19]:


N = int(input())
L=[];
for i in range(0,N):
    cmd=input().split();
    if cmd[0] == "insert":
        L.insert(int(cmd[1]),int(cmd[2]))
    elif cmd[0] == "append":
        L.append(int(cmd[1]))
    elif cmd[0] == "pop":
        L.pop();
    elif cmd[0] == "print":
        print(L)
    elif cmd[0] == "remove":
        L.remove(int(cmd[1]))
    elif cmd[0] == "sort":
        L.sort();
    else:
        L.reverse();


# ### 6. Tuples
Task
Given an integer, n, and n space-separated integers as input,t create a tuple, n, of those  integers. Then compute and print the result of hash(t).

Note: hash() is one of the functions in the __builtins__ module, so it need not be imported.

Input Format

The first line contains an integer, n, denoting the number of elements in the tuple.
The second line contains n space-separated integers describing the elements in tuple t.

Output Format

Print the result of hash(t).

Sample Input 0
2
1 2

Sample Output 0
3713081631934410656
# In[24]:


n = int(input())
int_list = [int(i) for i in input().split()]
int_tuple = tuple(int_list)
print(hash(int_tuple))


# ### 7. Introduction to Sets
A set is an unordered collection of elements without duplicate entries.
When printed, iterated or converted into a sequence, its elements will appear in an arbitrary order.

Example

>>> print set()
set([])

>>> print set('HackerRank')
set(['a', 'c', 'e', 'H', 'k', 'n', 'r', 'R'])

>>> print set([1,2,1,2,3,4,5,6,0,9,12,22,3])
set([0, 1, 2, 3, 4, 5, 6, 9, 12, 22])

>>> print set((1,2,3,4,5,5))
set([1, 2, 3, 4, 5])

>>> print set(set(['H','a','c','k','e','r','r','a','n','k']))
set(['a', 'c', 'r', 'e', 'H', 'k', 'n'])

>>> print set({'Hacker' : 'DOSHI', 'Rank' : 616 })
set(['Hacker', 'Rank'])

>>> print set(enumerate(['H','a','c','k','e','r','r','a','n','k']))
set([(6, 'r'), (7, 'a'), (3, 'k'), (4, 'e'), (5, 'r'), (9, 'k'), (2, 'c'), (0, 'H'), (1, 'a'), (8, 'n')])

Basically, sets are used for membership testing and eliminating duplicate entries.
# In[29]:


def average(array):
    
    array = set(array)
    return sum(array) / len(array)

if __name__ == '__main__':
    
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


# ### 8. No Idea!

# In[30]:


n,m = map(int,input().split())
N = list(map(int,input().split()))
A = set(map(int,input().split()))
B = set(map(int,input().split()))
#Union set A & B
C = A | B
#Exclude all numbers which doesn't exit in both A & B
N = (i for i in N if i in C)
#Add 1 if number is in set A else subtract 1
print(sum(1 if i in A else -1 for i in N ))


# ### 9. Symmetric Difference
Objective
Today, we're learning about a new data type: sets.

Concept

If the inputs are given on one line separated by a character (the delimiter), use split() to get the separate values in the form of a list. The delimiter is space (ascii 32) by default. To specify that comma is the delimiter, use string.split(','). For this challenge, and in general on HackerRank, space will be the delimiter.

Usage:

>> a = raw_input()
5 4 3 2
>> lis = a.split()
>> print (lis)
['5', '4', '3', '2']

If the list values are all integer types, use the map() method to convert all the strings to integers.

>> newlis = list(map(int, lis))
>> print (newlis)
[5, 4, 3, 2]

Sets are an unordered collection of unique values. A single set contains values of any immutable data type.

CREATING SETS

>> myset = {1, 2} # Directly assigning values to a set
>> myset = set()  # Initializing a set
>> myset = set(['a', 'b']) # Creating a set from a list
>> myset
{'a', 'b'}

MODIFYING SETS

Using the add() function:

>> myset.add('c')
>> myset
{'a', 'c', 'b'}
>> myset.add('a') # As 'a' already exists in the set, nothing happens
>> myset.add((5, 4))
>> myset
{'a', 'c', 'b', (5, 4)}

Using the update() function:

>> myset.update([1, 2, 3, 4]) # update() only works for iterable objects
>> myset
{'a', 1, 'c', 'b', 4, 2, (5, 4), 3}
>> myset.update({1, 7, 8})
>> myset
{'a', 1, 'c', 'b', 4, 7, 8, 2, (5, 4), 3}
>> myset.update({1, 6}, [5, 13])
>> myset
{'a', 1, 'c', 'b', 4, 5, 6, 7, 8, 2, (5, 4), 13, 3}

REMOVING ITEMS

Both the discard() and remove() functions take a single value as an argument and removes that value from the set. If that value is not present, discard() does nothing, but remove() will raise a KeyError exception.

>> myset.discard(10)
>> myset
{'a', 1, 'c', 'b', 4, 5, 7, 8, 2, 12, (5, 4), 13, 11, 3}
>> myset.remove(13)
>> myset
{'a', 1, 'c', 'b', 4, 5, 7, 8, 2, 12, (5, 4), 11, 3}

COMMON SET OPERATIONS Using union(), intersection() and difference() functions.

>> a = {2, 4, 5, 9}
>> b = {2, 4, 11, 12}
>> a.union(b) # Values which exist in a or b
{2, 4, 5, 9, 11, 12}
>> a.intersection(b) # Values which exist in a and b
{2, 4}
>> a.difference(b) # Values which exist in a but not in b
{9, 5}

The union() and intersection() functions are symmetric methods:

>> a.union(b) == b.union(a)
True
>> a.intersection(b) == b.intersection(a)
True
>> a.difference(b) == b.difference(a)
False

These other built-in data structures in Python are also useful.
# In[32]:


input()
m = set(map(int,input().split()))
input()
n = set(map(int,input().split()))
print(*sorted(m^n), sep="\n")


# ### 10. Set .add()
# 
# 
If we want to add a single element to an existing set, we can use the .add() operation.
It adds the element to the set and returns 'None'.
# In[2]:


N = int(input("Enter the total number of country stamps : "))

countries = set()

for i in range(N):
    countries.add(input("Enter the name of the country where the stamp is from : "))

print("The total number of distinct country stamps : ",len(countries))


# ### 11. Set .discard(), .remove() & .pop()
remove(x)
This operation removes element x from the set.
If element x does not exist, it raises a KeyError.
The .remove(x) operation returns None.
# In[4]:


num = int(input("Enter the number of elements in the set : "))
data = set(map(int, input("Enter space separated elements of set : ").split()))
operations = int(input("Enter the number of commands : "))

for x in range(operations):
  oper = input("Enter either pop, remove and/or discard commands followed by their associated value : ").split()
  if oper[0] == "remove":
    data.remove(int(oper[1]))
  elif oper[0] == "discard":
    data.discard(int(oper[1]))
  else:
    data.pop()
    
print(sum(data))


# ### 12. Set .union() Operation
.union()

The .union() operator returns the union of a set and the set of elements in an iterable.
Sometimes, the | operator is used in place of .union() operator, but it operates only on the set of elements in set.
Set is immutable to the .union() operation (or | operation).
# In[7]:


N1 = int(input("Enter the number of students who have subscribed to the English newspaper : "))
English = set(input("Enter space separated roll numbers of those students : ").split());

N2 = int(input("Enter the number of students who have subscribed to the French newspaper : "))
French = set(input("Enter space separated roll numbers of those students : ").split());

Union = English.union(French)

print("Total number of students who have at least one subscription : ", len(Union))


# ### 13. Set .intersection() Operation
.intersection()
The .intersection() operator returns the intersection of a set and the set of elements in an iterable.
Sometimes, the & operator is used in place of the .intersection() operator, but it only operates on the set of elements in set.
The set is immutable to the .intersection() operation (or & operation).
# In[10]:


N1 = int(input("Enter the number of students who have subscribed to the English newspaper : "))
English = set(input("Enter space separated roll numbers of those students : ").split())

N2 = int(input("Enter the number of students who have subscribed to the French newspaper : "))
French = set(input("Enter space separated roll numbers of those students : ").split())

Both = French.intersection(English)

print("Total number of students who have subscriptions to both English and French newspapers : ", len(Both))


# ### 14. Set .difference() Operation
# 
# 
.difference()
The tool .difference() returns a set with all the elements from the set that are not in an iterable.
Sometimes the - operator is used in place of the .difference() tool, but it only operates on the set of elements in set.
Set is immutable to the .difference() operation (or the - operation).
# In[12]:


N1 = int(input("Enter the number of students who have subscribed to the English newspaper : "))
English = set(input("Enter space separated roll numbers of those students : ").split())

N2 = int(input("Enter the number of students who have subscribed to the French newspaper : "))
French = set(input("Enter space separated roll numbers of those students : ").split())

Only_English = English.difference(French)

print("Total number of students who are subscribed to the English newspaper only : ", len(Only_English))


# ### 15. Set .symmetric_difference() Operation
.symmetric_difference()
The .symmetric_difference() operator returns a set with all the elements that are in the set and the iterable but not both.
Sometimes, a ^ operator is used in place of the .symmetric_difference() tool, but it only operates on the set of elements in set.
The set is immutable to the .symmetric_difference() operation (or ^ operation).
# In[14]:


N1 = int(input("Enter the number of students who have subscribed to the English newspaper : "))
English = set(input("Enter space separated roll numbers of those students : ").split())

N2 = int(input("Enter the number of students who have subscribed to the French newspaper : "))
French = set(input("Enter space separated roll numbers of those students : ").split())

Either = English.symmetric_difference(French)

print("Total number of students who have subscriptions to the English or the French newspaper but not both : ", len(Either))


# ### 16. Set Mutations
We have seen the applications of union, intersection, difference and symmetric difference operations, but these operations do not make any changes or mutations to the set.

We can use the following operations to create mutations to a set:

.update() or |=
Update the set by adding elements from an iterable/another set.

.intersection_update() or &=
Update the set by keeping only the elements found in it and an iterable/another set.

.difference_update() or -=
Update the set by removing elements found in an iterable/another set.

.symmetric_difference_update() or ^=
Update the set by only keeping the elements found in either set, but not in both.
# In[16]:


len_set_A = int(input("Enter the number of elements in set A : "))

set_A = set(map(int, input("Enter space separated list of elements in set A : ").split()))

len_other_set = int(input("Enter the number of other sets : "))

for i in range(len_other_set):
    operation = input("Enter space separated operation name and the length of the other set : ").split()
    if operation[0] == 'intersection_update':
        temp_storage = set(map(int, input("Enter space separated list of elements in the other set : ").split()))
        set_A.intersection_update(temp_storage)
    elif operation[0] == 'update':
        temp_storage = set(map(int, input("Enter space separated list of elements in the other set : ").split()))
        set_A.update(temp_storage)
    elif operation[0] == 'symmetric_difference_update':
        temp_storage = set(map(int, input("Enter space separated list of elements in the other set : ").split()))
        set_A.symmetric_difference_update(temp_storage)
    elif operation[0] == 'difference_update':
        temp_storage = set(map(int, input("Enter space separated list of elements in the other set : ").split()))
        set_A.difference_update(temp_storage)
    else :
        assert False

print(sum(set_A))


# ### 17. The Captain's Room
# 
# 
Mr. Anant Asankhya is the manager at the INFINITE hotel. The hotel has an infinite amount of rooms.

One fine day, a finite number of tourists come to stay at the hotel.

The tourists consist of:
→ A Captain.
→ An unknown group of families consisting of K  members per group where  K!=1.

The Captain was given a separate room, and the rest were given one room per group.

Mr. Anant has an unordered list of randomly arranged room entries. The list consists of the room numbers for all of the tourists. The room numbers will appear K  times per group except for the Captain's room.

Mr. Anant needs you to help him find the Captain's room number.
The total number of tourists or the total number of groups of families is not known to you.
You only know the value of K  and the room number list.
# In[17]:


K = int(input("Enter the size of each group : "))

room_no_list = map(int, input("Enter the unordered elements of the room number list : ").split())
room_no_list = sorted(room_no_list)

for i in range(len(room_no_list)):
    if(i != len(room_no_list)-1):
        if(room_no_list[i] != room_no_list[i-1] and room_no_list[i] != room_no_list[i+1]):
            print(room_no_list[i])
            break;
    else:
        print(room_no_list[i])


# ### 18. Check Subset
# 
# 
You are given two sets, A and B.
Your job is to find whether set A is a subset of set B.

If set A is subset of set B, print True.
If set A is not a subset of set B, print False.
# In[18]:


T = int(input("Enter the number of test cases, T : "))

for i in range(T):
    a = input("Enter the number of elements in set A : ")
    A = set(input("Enter the space separated elements of set A : ").split())
    b = int(input("Enter the number of elements in set B : "))
    B = set(input("Enter the space separated elements of set A : ").split())
    print(A.issubset(B))


# ### Check Strict Superset
You are given a set A and n other sets.
Your job is to find whether set A is a strict superset of each of the  sets.

Print True, if A is a strict superset of each of the N sets. Otherwise, print False.

A strict superset has at least one element that does not exist in its subset.
# In[19]:


A_set = set(input("Enter the space separated elements of set A : ").split())
N = int(input("Enter the number of other sets : "))
output = True

for i in range(N):
    other_set = set(input("Enter the space separated elements of the other sets : ").split())
    if not other_set.issubset(A_set):
        output = False
    if len(other_set) >= len(A_set):
        output = False

print(output)


# In[ ]:




