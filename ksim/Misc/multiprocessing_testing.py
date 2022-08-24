# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:07:37 2020

@author: BVH
"""
import multiprocessing as mp

import numpy as np

    


def my_func(x,y):
  #print(mp.current_process())
  print(y)
  return x**x

def main():
  pool = mp.Pool(mp.cpu_count())
  result = pool.starmap(my_func, [[4,3],[5,np.array([32,53])]])
  result_set_2 = pool.starmap(my_func, [[4,5],[5,4]])

  print(result)
  print(result_set_2)




if __name__ == "__main__":
    print('hello')
    main()