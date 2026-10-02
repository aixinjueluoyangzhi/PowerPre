# 基金持仓对账助手 README

## 1. 项目简介

本项目实现一个基金持仓对账助手，用于每日收市后比对：

- **清算文件持仓**（`CLEARING`）
- **内部系统持仓**（`SYSTEM`）

找出所有真实差异，生成结构化对账报告，并对每条真实差异调用大模型生成归因分析与建议处理方式。

核心原则：

- 差异判定必须由确定性代码完成，跑两次结果一致。
- 大模型只负责归因分析，不参与差异判定。
- 两边金额单位不同，统一后再比较。
- 市值差异小于 1 元视为精度误差，不进报告。
- 基金匹配以 `code` 为准，`name` 不参与匹配。
- 只在一边存在的记录单独列出。
- LLM 调用失败时降级，不阻断对账。

---

## 2. 环境要求

- Python 3.8+
- `requests`

安装依赖：

```bash
pip install requests
```

---

## 3. 怎么跑

### 3.1 配置 API Key

打开 `编程题F-持仓对账.py`，修改顶部三行：

```python
BASE_URL = "https://api.deepseek.com"   # 换成你用的 base_url
API_KEY  = ""                            # 填入你的 API key
MODEL    = "deepseek-flash"              # 换成你用的模型
```

> 提交前请删除 `API_KEY`，不要泄露密钥。

### 3.2 运行

```bash
python3 编程题F-持仓对账.py
```

### 3.3 有 API Key 时

- 对每条双边真实差异调用一次 LLM。
- 对每条单边记录调用一次 LLM。
- 归因写入 `attribution` 字段。

### 3.4 无 API Key 时

脚本会自动检测 `API_KEY` 是否为空：

- 为空：跳过 LLM 调用，`attribution` 填 `error: "LLM 归因已禁用"`。
- 不为空：正常调用 LLM。

对账结果本身与是否调用 LLM 无关，保证确定性部分始终可运行。

---

## 4. 输出结构

`reconcile()` 返回结构化字典，主要字段：

```json
{
  "stage": "final",
  "summary": {
    "clearing_total": 5,
    "system_total": 4,
    "matched_count": 4,
    "passed_count": 2,
    "diff_count": 2,
    "only_in_clearing_count": 1,
    "only_in_system_count": 0,
    "market_value_diff_threshold_yuan": 1.0,
    "wan_to_yuan": 10000.0,
    "llm_enabled": true
  },
  "differences": [],
  "only_in_clearing": [],
  "only_in_system": []
}
```

### 4.1 `summary`

对账汇总信息：

- `clearing_total`：清算文件记录数
- `system_total`：内部系统记录数
- `matched_count`：两边 code 都能匹配上的数量
- `passed_count`：匹配上且无真实差异的数量
- `diff_count`：双边真实差异数量
- `only_in_clearing_count`：只在清算文件的记录数
- `only_in_system_count`：只在内部系统的记录数
- `market_value_diff_threshold_yuan`：市值差异阈值，1 元
- `wan_to_yuan`：万元转元换算系数
- `llm_enabled`：本次是否启用了 LLM 归因

### 4.2 `differences`

双边真实差异列表，每条包含：

- `code`
- `type`: `BILATERAL_DIFF`
- `diff_types`: `SHARES_DIFF` / `MARKET_VALUE_DIFF`
- `clearing`: 清算侧名称、份额、市值（元）
- `system`: 系统侧名称、份额、市值（元）
- `shares_diff`: 系统份额 - 清算份额
- `market_value_diff`: 系统市值 - 清算市值
- `shares_diff_abs`
- `market_value_diff_abs`
- `threshold_yuan`
- `attribution`: LLM 归因分析

### 4.3 `only_in_clearing`

只在清算文件存在的记录：

- `code`
- `type`: `ONLY_IN_CLEARING`
- `side`: `CLEARING`
- `name`
- `shares`
- `market_value_yuan`
- `attribution`

### 4.4 `only_in_system`

只在内部系统存在的记录，结构同上，`type` 为 `ONLY_IN_SYSTEM`。

### 4.5 `attribution`

LLM 归因结构：

```json
{
  "possible_reason": "可能原因",
  "suggested_action": "建议处理方式",
  "raw": "LLM 原始返回文本",
  "error": null
}
```

失败时：

```json
{
  "possible_reason": null,
  "suggested_action": null,
  "raw": null,
  "error": "LLM 调用失败: ..."
}
```

---

## 5. 设计思路

### 5.1 整体流程

1. **数据标准化**
   - 清算文件市值单位：万元 → 元
   - 内部系统市值单位：元 → 保持不变
   - 统一字段名，保留原始记录便于追溯
   - `code` 一律按字符串处理，保留前导零

2. **按 code 建索引**
   - `clearing_by_code`
   - `system_by_code`
   - 同时记录重复 code 冲突

3. **集合对账**
   - 两边 code 交集：双边记录
   - 只在清算：单边记录
   - 只在系统：单边记录

4. **差异判定**
   - 双边记录：
     - 份额差异：只要存在就算真实差异
     - 市值差异：绝对值 >= 1 元才算真实差异
   - 市值差异 < 1 元且份额一致：视为通过

5. **差异分类**
   - `SHARES_DIFF`
   - `MARKET_VALUE_DIFF`
   - 两者可同时出现

6. **LLM 归因**
   - 对每条真实差异构造 prompt
   - 调用 `call_llm`
   - 解析为结构化 JSON
   - 失败时降级，不阻断对账

7. **输出报告**
   - 所有列表按 `code` 排序
   - 保证确定性部分可复现

### 5.2 为什么用 code 而不是 name

题目明确要求：基金匹配以代码为准，名字很像的可能是两只不同的基金。

本实现中：

- `012345` 清算名“华信成长混合A”，系统名“华信成长混合”，视为同一基金。
- `008872` 清算名“远见科技先锋”，系统名“远见科技先锋混合”，视为同一基金。
- `name` 只用于展示，不参与匹配。

### 5.3 为什么市值差异阈值是 1 元

题目要求：市值差异小于 1 元的属于精度误差，不算差异，不进报告。

本实现中：

- `008872` 市值差 0.40 元，< 1 元，未进差异列表。
- `012346` 市值差 5 元，>= 1 元，进差异列表。

注意：

- 阈值只作用于市值差异。
- 份额差异只要存在就算真实差异，不受 1 元阈值影响。

### 5.4 为什么 LLM 不参与差异判定

题目硬性要求：

- 数值比对必须用确定性代码实现，跑两次结果要一模一样。
- 把两张表整个丢给大模型让它找差异，不算完成。

因此本实现：

- 差异判定、分类、汇总全部由代码完成。
- LLM 只接收“已经判定出的差异”，生成归因分析和建议。
- LLM 输出不反向影响差异事实。

### 5.5 失败降级

LLM 调用可能因以下原因失败：

- 网络错误
- 超时
- 限流
- 返回非 JSON

本实现：

- 捕获所有异常。
- 在 `attribution.error` 中记录错误原因。
- 对账结果本身不受影响。
- 报告仍可正常生成。

---

## 6. 测试用例与预期结果

使用题目给定数据，预期结果：

### 6.1 汇总

- `clearing_total = 5`
- `system_total = 4`
- `matched_count = 4`
- `passed_count = 2`
- `diff_count = 2`
- `only_in_clearing_count = 1`
- `only_in_system_count = 0`

### 6.2 真实差异

1. `012346`
   - 类型：`MARKET_VALUE_DIFF`
   - 份额一致
   - 市值差 5 元

2. `015800`
   - 类型：`SHARES_DIFF`
   - 份额差 250,000
   - 市值一致

### 6.3 单边记录

- `021100` 只在清算文件。

### 6.4 通过记录

- `012345`：份额一致，市值差 0
- `008872`：份额一致，市值差 0.40 元 < 1 元

---

## 7. 已知缺陷

1. **LLM 归因内容不稳定**
   - 同一差异多次调用，返回文本可能不同。
   - 但差异判定本身不受影响，确定性部分可复现。

2. **LLM 返回解析容错有限**
   - 当前支持直接 JSON 和 markdown code fence 中的 JSON。
   - 如果模型返回其他格式，会降级为 `raw` 文本，`possible_reason` 和 `suggested_action` 为 `null`。

3. **未处理重复 code**
   - 代码中已检测重复 code 并记录到 `clearing_duplicates` / `system_duplicates`。
   - 但当前对账逻辑保留第一条，未实现更复杂的冲突合并策略。

4. **未实现金额格式化**
   - 当前输出为浮点数。
   - 给运营看的报告可进一步格式化为千分位、保留两位小数。

5. **未实现并发调用 LLM**
   - 当前逐条串行调用。
   - 差异较多时耗时较长，可改为并发，但需注意限流。

6. **未实现命令行参数**
   - 当前通过 `API_KEY` 是否为空决定是否启用 LLM。
   - 可增加 `--no-llm` 参数，便于测试。

7. **未实现日志**
   - 当前直接打印 JSON。
   - 生产环境建议增加日志、错误监控和重试机制。

---