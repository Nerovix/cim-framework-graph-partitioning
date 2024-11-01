from sortedcontainers import SortedSet

class prefix_nodeset():
    def __init__(self):
        self.intervals = SortedSet() 

    def add(self, num):
        
        new_interval = (num, num) 
        index = self.intervals.bisect_right((num+1,num))
        
        if index > 0 and self.intervals[index - 1][1] >= num:
            return
        
        if index > 0 and self.intervals[index - 1][1] == num - 1:
            new_interval=(self.intervals[index - 1][0], new_interval[1])
            self.intervals.pop(index-1)
            index-=1

        if index < len(self.intervals) and self.intervals[index][0]==num+1:
            new_interval=(new_interval[0],self.intervals[index][1])
            self.intervals.pop(index)
        
        self.intervals.add(new_interval) 

    def remove(self, num):
        index = self.intervals.bisect_right((num+1, num))
        if index > 0 and self.intervals[index - 1][1] >= num:
            (L,R)=self.intervals[index - 1]
            self.intervals.pop(index-1)
            if L<=num-1:
                self.intervals.add((L,num-1))
            if num+1<=R:
                self.intervals.add((num+1,R))
    def have(self, num):
        index = self.intervals.bisect_right((num+1, num))
        if index > 0 and self.intervals[index - 1][1] >= num:
            return True
        return False

    def __eq__(self, other):
        if isinstance(other, prefix_nodeset):
            return self.intervals == other.intervals
        return False

    def __hash__(self):
        return hash(tuple(self.intervals))

    def __repr__(self):
        return f"prefix_nodeset({list(self.intervals)})"

# test
if __name__ == "__main__":
    interval_set = prefix_nodeset()
    interval_set.add(1)  
    interval_set.add(3) 
    interval_set.add(2) 
    interval_set.add(5)  
    print(interval_set)  
    print(interval_set.have(2)) 
    print(interval_set.have(4)) 
    interval_set.remove(3) 
    print(interval_set) 
    interval_set.remove(1) 
    print(interval_set) 
