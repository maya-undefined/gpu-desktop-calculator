# gdc

need to add a column of numbers an exabyte in size from the cli?


-rw-rw-r--  1 user user 880577984 Nov 18 16:42 f2.dat
-rw-rw-r--  1 user user 880577984 Nov 18 16:42 f1.dat

```
native:
	chunk size:   5 * 1024 * 1024		6m45s
gpu-calcualtor: 
	chunk size:   5 * 1024 * 1024		6m50s :(
	chunk size: 100 * 1024 * 1024		cored
```