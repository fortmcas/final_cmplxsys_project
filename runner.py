import subprocess
import multiprocessing
success_event = multiprocessing.Event()
success_event.clear()

def run_my_garbage():
    global success_event
    success_value = -1000
    while success_value < -0.1:
        subprocess.run(['python3', 'neural_net.py'])
        success_value = float(subprocess.check_output(['python3', 'evaluate_after_training.py']))
    success_event.set()

if __name__ == "__main__":
    processes = []
    for i in range(8):
        processes.append(multiprocessing.Process(target=run_my_garbage))
    for i in processes:
        i.start()
    success_event.wait()
    for i in processes:
        i.kill()
        i.join()
