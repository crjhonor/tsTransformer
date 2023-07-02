"""
Activation
"""
from EmQuantAPI import *
import platform
#手动激活范例(单独使用)
#获取当前安装版本为x86还是x64

"""
data = platform.architecture()

if data[0] == "64bit":
    bit = "x64"
elif data[0] == "32bit":
    bit = "x86"

data1 = platform.system()
if data1 == 'Linux':
    system1 = 'linux'
    lj = c.setserverlistdir("libs/" + system1 + '/' + bit)
elif data1 == 'Windows':
    system1 = 'windows'
    lj = c.setserverlistdir("libs/" + system1)
elif data1 == 'Darwin':
    system1 = 'mac'
    lj = c.setserverlistdir("libs/" + system1)
else:
    pass
#调用manualactive函数，修改账号、密码、有效邮箱地址，email=字样需保留
data = c.manualactivate("18666712666", "jay275699", "email=1012933333@qq.com")
if data.ErrorCode != 0:
    print ("manualactivate failed, ", data.ErrorMsg)
"""

loginresult = c.start()

data = c.css("TA0.CZC", "CLOSE", "TradeDate=2023-06-29")
if data.ErrorCode != 0:
    print(f"request css Error, {data.ErrorMsg}")
else:
    for code in data.Codes:
        for i in range(0,len(data.Indicators)):
            print(data.Data[code][i])

loginresult = c.stop()

print("Done")