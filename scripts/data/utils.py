IMPOSSIBLE_TASKS = [
    "图",
    "导出",
    "短信",
    "邮件",
    "加载","保存",
]
RUBBISH_CODE_COLLECTIONS = [
    "const sheet = Application.ActiveSheet;\nconst usedRange = sheet.UsedRange;\nconst rowCount = usedRange.Rows.Count;",
    "const sheet = Application.ActiveSheet\nconst usedRange = sheet.UsedRange\nconst rowCount = usedRange.Rows.Count",
    "const sheet = Application.ActiveSheet\nconst usedRange = sheet.UsedRange\nconst rowCount = usedRange.Rows.Count\nfor(let i = usedRange.Row + 1; i <= rowCount; i++) {\n}",
]

def longest_common_substring(s1, s2):
    m = len(s1)
    n = len(s2)

    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Result storing variables
    max_len = 0
    longest_substr = ""

    # DP logic
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                if dp[i + 1][j + 1] > max_len:
                    max_len = dp[i + 1][j + 1]
                    longest_substr = s1[i - max_len + 1:i + 1]

    return longest_substr