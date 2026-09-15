# Meta-atom 資料庫

此資料夾集中存放 asia_1290.npy、asia_1310.npy、asia_1330.npy。
檔名數字代表真空波長（nm）。資料庫內幾何參數使用 nm，
phase 使用 rad；transmission 的振幅/強度定義仍須由來源確認。

NPY 含 pickle 物件，只應載入可信來源。範例與 UI 預設從此處載入，
波長分析仍可透過 --library-dir 指定其他資料庫資料夾。
