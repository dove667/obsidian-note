## **1.** `**str**`**（字符串）常用方法**

```Python
s = "hello world"
```

|   |   |   |
|---|---|---|
|方法|作用|示例|
|`s.upper()`|转大写|`'HELLO WORLD'`|
|`s.lower()`|转小写|`'hello world'`|
|`s.capitalize()`|首字母大写|`'Hello world'`|
|`s.title()`|每个单词首字母大写|`'Hello World'`|
|`s.strip()`|去除首尾空格/指定字符|`'hello world'`|
|`s.lstrip()`|去除左侧空格|`'hello world'`|
|`s.rstrip()`|去除右侧空格|`'hello world'`|
|`s.replace("old", "new")`|替换字符串|`'hello Python'`|
|`s.split(" ")`|按分隔符拆分成列表|`['hello', 'world']`|
|`s.join(iterable)`|用 `s` 连接 `iterable`|`"-".join(['hello', 'world']) → 'hello-world'`|
|`s.find("o")`|查找子串索引（找不到返回 `-1`）|`4`|
|`s.index("o")`|查找子串索引（找不到抛 `ValueError`）|`4`|
|`s.count("o")`|统计子串出现次数|`2`|
|`s.startswith("h")`|是否以某子串开头|`True`|
|`s.endswith("d")`|是否以某子串结尾|`True`|
|`s.isdigit()`|是否全是数字|`False`|
|`s.isalpha()`|是否全是字母|`False`|
|`s.isalnum()`|是否全是字母或数字|`False`|
|`s.isspace()`|是否全是空格|`False`|
|`s.swapcase()`|大小写互换|`'HELLO WORLD'`|
|`s.zfill(10)`|左侧补 `0` 使总长度为 `10`|`'000hello world'`|

---

## **2.** `**list**`**（列表）常用方法**

```Python
lst = [1, 2, 3, 4]
```

|   |   |   |
|---|---|---|
|方法|作用|示例|
|`lst.append(x)`|追加元素到末尾|`[1, 2, 3, 4, 5]`|
|`lst.extend(iterable)`|追加可迭代对象|`[1, 2, 3, 4, 5, 6]`|
|`lst.insert(i, x)`|在索引 `i` 位置插入元素 `x`|`[1, 2, 99, 3, 4]`|
|`lst.remove(x)`|删除第一个值等于 `x` 的元素|`[1, 3, 4]`|
|`lst.pop(i=-1)`|删除索引 `i` 位置元素（默认删除最后一个）|`[1, 2, 3]`|
|`lst.index(x)`|查找 `x` 的索引（找不到抛 `ValueError`）|`2`|
|`lst.count(x)`|统计 `x` 出现次数|`1`|
|`lst.sort()`|原地升序排序|`[1, 2, 3, 4]`|
|`lst.sort(reverse=True)`|原地降序排序|`[4, 3, 2, 1]`|
|`sorted(lst)`|生成新排序列表（不会修改原列表）|`[1, 2, 3, 4]`|
|`lst.reverse()`|原地翻转列表|`[4, 3, 2, 1]`|
|`reversed(lst)`|生成新翻转迭代器|`[4, 3, 2, 1]`|
|`lst.copy()`|复制列表|`[1, 2, 3, 4]`|
|`lst.clear()`|清空列表|`[]`|

---

## **3.** `**tuple**`**（元组）常用方法**

```Python
tup = (1, 2, 3, 2)
```

|   |   |   |
|---|---|---|
|方法|作用|示例|
|`tup.count(x)`|统计 `x` 出现次数|`2`|
|`tup.index(x)`|查找 `x` 第一次出现的索引|`1`|
|`len(tup)`|计算元组长度|`4`|

📌 **注意**：

- **元组是不可变的**，没有 `append()`、`remove()` 这类方法。
- **要修改元组，必须创建新元组**：
    
    ```Python
    python
    复制编辑
    new_tup = tup + (4, 5)  # (1, 2, 3, 2, 4, 5)
    
    ```
    

---

## **4.** `**dict**`**（字典）常用方法**

```Python
d = {"a": 1, "b": 2, "c": 3}
```

|   |   |   |
|---|---|---|
|方法|作用|示例|
|`d.keys()`|获取所有键|`dict_keys(['a', 'b', 'c'])`|
|`d.values()`|获取所有值|`dict_values([1, 2, 3])`|
|`d.items()`|获取所有键值对|`dict_items([('a', 1), ('b', 2), ('c', 3)])`|
|`d.get(key, default)`|获取 `key` 对应值，不存在返回 `default`|`2`|
|`d.pop(key, default)`|删除 `key` 并返回其值|`d.pop("b") → 2`|
|`d.popitem()`|删除并返回**最后一个**键值对（Python 3.7+）|`("c", 3)`|
|`d.setdefault(key, default)`|获取 `key` 的值，不存在则设置默认值|`d.setdefault("d", 4) → 4`|
|`d.update(other_dict)`|合并 `other_dict` 到 `d`|`{'a': 1, 'b': 2, 'c': 3, 'd': 4}`|
|`d.clear()`|清空字典|`{}`|
|`d.copy()`|复制字典|`{"a": 1, "b": 2, "c": 3}`|

---

## **5. 额外补充**

### `**set**`**（集合）常用方法**

```Python
s = {1, 2, 3}

```

|   |   |   |
|---|---|---|
|方法|作用|示例|
|`s.add(x)`|添加元素 `x`|`{1, 2, 3, 4}`|
|`s.remove(x)`|删除元素 `x`（不存在会报错）|`{1, 3}`|
|`s.discard(x)`|删除元素 `x`（不存在不会报错）|`{1, 3}`|
|`s.pop()`|随机删除并返回一个元素|`1`|
|`s.union(other)`|并集|`{1, 2, 3, 4, 5}`|
|`s.intersection(other)`|交集|`{2, 3}`|
|`s.difference(other)`|差集|`{1}`|
|`s.symmetric_difference(other)`|对称差集|`{1, 4, 5}`|

---

### **总结**

✅ **字符串 (**`**str**`**)** → 主要用于**文本处理**。

✅ **列表 (**`**list**`**)** → **可变**，常用于**存储有序数据**。

✅ **元组 (**`**tuple**`**)** → **不可变**，数据不能修改，性能比 `list` 更高。

✅ **字典 (**`**dict**`**)** → **键值对映射**，适用于**快速查找数据**。

✅ **集合 (**`**set**`**)** → **去重、数学运算（并交差集）**。

🚀 **熟练掌握这些方法，写 Python 代码更高效！**