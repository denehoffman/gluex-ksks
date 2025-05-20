from modak import TaskQueue

from gluex_ksks.constants import LOG_PATH
from gluex_ksks.tasks.cuts import ChiSqDOF

if __name__ == '__main__':
    LOG_PATH.mkdir(exist_ok=True)
    tq = TaskQueue(workers=4, state_file_path=LOG_PATH)
    tq.run([ChiSqDOF(data_type='data', run_period='s17', chisqdof=3.4)])
