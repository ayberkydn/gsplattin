#!/bin/bash
#SBATCH -p palamut-cuda     # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -o out.txt    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 2   # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=4:00:00      # Sure siniri koyun.

# Çalıştırılacak komutlar
bash runagents.sh &
bash runagents.sh &
bash runagents.sh &

wait
