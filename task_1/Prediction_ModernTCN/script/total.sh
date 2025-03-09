#!/bin/bash

nohup bash dpw_nanao.sh > dpw_nanao_2021_2022.log 2>&1 &
nohup bash dpw_dameisha.sh > dpw_dameisha_2021_2022.log 2>&1 &
nohup bash dpw_stj.sh > dpw_stj_2021_2022.log 2>&1 &
nohup bash dpw_wankou.sh > dpw_wankou_2021_2022.log 2>&1 &
nohup bash dpw_xs.sh > dpw_xs_2021_2022.log 2>&1 &

nohup bash dyw_baguang.sh > dyw_baguang_2021_2022.log 2>&1 &
nohup bash dyw_changwan.sh > dyw_changwan_2021_2022.log 2>&1 &
nohup bash dyw_dongshan.sh > dyw_dongshan_2021_2022.log 2>&1 &
nohup bash dyw_dongyong.sh > dyw_dongyong_2021_2022.log 2>&1 &

nohup bash szw_sk.sh > szw_sk_2021_2022.log 2>&1 &

nohup bash zjk_fs.sh > zjk_fs_2021_2022.log 2>&1 &
nohup bash zjk_ln.sh > zjk_ln_2021_2022.log 2>&1 &

nohup bash train_M.sh > train_M.log 2>&1 &