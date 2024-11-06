class Sequence (object):#Base class
    def __init__ (self, array):
        self.array = array
        
    def __iter__(self):#Method called when iterating
        return Iterator(self)
    
    def __len__(self):#Method called when using len()
        return len(self.array)
    
    def __eq__(self, other):#Method called when using the operator "=="
        if len(self) != len(other):
            raise ValueError("Two arrays are not equal in length !")
        equal_count = sum(1 for a, b in zip(self, other) if a == b)
        return equal_count
    
class Iterator:
    def __init__(self, seq):
        self.items = seq.array
        self.index = -1
    def __iter__(self):
        return self
    def __next__(self):#Iterate by increasing the value of "self.index" and accessing that position in the array
        self.index += 1
        if self.index < len(self.items):
            return self.items[self.index]
        else:
            raise StopIteration

class Arithmetic (Sequence):#Subclass of Sequence
    def __init__(self, start, step):
        self.start = start
        self.step = step
        
    def __call__(self, length):
        self.array = list(range(self.start, self.start+length*self.step, self.step)) #Create array of length "length" of an arithmetic sequence starting from "self.start" with step "self.step"
        #print(self.array)
        
class Geometric (Sequence):#Subclass of sequence
    def __init__(self, start, ratio):
        self.start = start
        self.ratio = ratio
        
    def __call__(self, length):
        self.array = [self.start * (self.ratio ** i) for i in range(length)] #Create array of length "length" of an geometric sequence starting from "self.start" with ratio "self.ratio"
        #print(self.array)
        

#Code to prove that the classes are well implemented        
AS = Arithmetic ( start =1 , step =2 )
AS( length =5 ) # [1, 3, 5, 7, 9]
print([n for n in AS]) 

GS = Geometric ( start =1 , ratio =2 )
GS( length =5 ) # [1, 2, 4, 8, 16]
print([n for n in GS]) 

print(AS == GS ) # 1

GS( length =8 ) # [1, 2, 4, 8, 16 , 32 , 64 , 128]
print([n for n in GS]) 

print( AS == GS ) # will raise an error