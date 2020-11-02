<p style="text-align:right;">
姓名:李韋宗<br>
學號:B10615024<br>
日期:2020/11/2<br>
</p>

<h1 style="text-align:center;"> Homework 2: Best Match Model

## 建置環境與套件
* Python 3.6.9, numpy, collections.Counter, tqdm.tqdm, datetime.datetime, timezone, timedelta, functools.reduce, argparse.ArgumentParser, ArgumentDefaultsHelpFormatter

## 資料前處理
* 按照 doc_list.txt/query_list.txt 的順序讀入文章及 query
* 將所有文章及 query 存成 list of string 方便後續操作
    * 並且集中 file I/O 的時間
    * 同時用 `Counter` 計算字典型態的 TF
* 透過 TF 計算字典型態的 DF

## 模型參數調整
* TF 使用出現文字出現在文章中的次數
* IDF 與簡報上的不同，除根據上次的經驗在 log 中加 1 外，還額外對 IDF 平方(讓 IDF 更為重要)
    * $IDF = (ln(1 + \frac{N+0.5}{n_i+0.5}))^2$
* $b = 0.75$
* $K_1 = 3.5$
* 不使用 query 的 TF 項(有 $K_3$ 的那項)

## 模型運作原理
* 為加速運算(偷吃步)，將單字表壓縮在 query 所出現的字
* 讓 TF, IDF 的維度都剩下 123 維
* 忽略計算相似度時的 query TF 項，即
    * $\frac{(K_3+1) \times tf_{i,q}}{K_3 \times tf_{i,q}} = 1$

## 心得
* 一開始無法突破 baseline 時就先嘗試各種 K1 值的曲線，但得分最高的組合仍然差 bseline 有將近 4 分，B 的調整無法有任何改進，而不調整 K3 因為根據多數網路上的資料，K3 的值都很不小，表是該項幾乎都接近 1，調整的顯著性應該不高，經過實驗後有無計算該項分數也都沒有差異，所以該項最後也直接刪除。刪除後有 0.6 分的提高，但仍然不及 baseline，於是上網找尋套件的幫助，但使用實驗下最好的 K1 值，套件也無法超過 baseline。於是更改文章 TF 的算法，從原先 `string.count()` 改為 `Counter()`，發現算出的 TF 不同且分數將近 3 分的進步，為了確定到底哪個算法才是對的，於是回頭更改作業 1(原先也使用`string.count()`計算)，發現分數也提高了，雖然在這次的作業依舊沒突破 baseline。研究發現可能的原因是 query 中的 123 個字或許有某些字為 substring，導致 `count()` 重複計算，而 `Counter()` 不會有這樣的問題，所以算出來的 TF 才是正確的。最後，根據上次的經驗，修改 IDF 的算法可能有辦法提升效能，於是對該項平方有了 1.2 分的進步，也成功超過 baseline。
* 附圖為上述實驗的正確率曲線(黑色為 baseline)
    * 修改前: 藍色為最初成績，刪除 query TF 項後為灰色，接著將 IDF 平方為淺藍色星點；黃色為套件 rank_bm25
    * 修改後: 棕線與紫色方點分別為有無刪除 query TF 的比較(IDF 沒有平方)；紅色為最終結果
![](https://i.imgur.com/ttpdSOY.png)


## 參考資料
> rank_bm25
>> https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py

> 參數選擇
>> https://www.elastic.co/blog/practical-bm25-part-3-considerations-for-picking-b-and-k1-in-elasticsearch

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
<!-- <img src="https://latex.codecogs.com/gif.latex?[formula]"/> -->