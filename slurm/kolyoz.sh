#!/bin/bash
#SBATCH -p kolyoz-cuda      # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -o out_%j_%a.txt    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 16  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=4:00:00      # Sure siniri koyun.
#SBATCH --array=0-3%4       # 4 parallel agents
#SBATCH --job-name=gsplat-%a

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Usage: $0 <wandb-agent-id>" >&2
  exit 1
fi

bash runagent.sh "$1"
