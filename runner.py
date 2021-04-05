import subprocess
success_value = -1000
while success_value < -0.1:
    subprocess.run(['python3', 'neural_net.py'])
    success_value = float(subprocess.check_output(['python3', 'evaluate_after_training.py']))
