# testing the data_processing file

from data_process import data_prep_task

a, b, c, d =data_prep_task('wash', test_size=0.3)
print(a.dtype)