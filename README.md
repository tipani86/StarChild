# Children of the Stars 星星的孩子

## Background 项目背景

Autistic children are a group that is often overlooked. They have introverted characteristics, but they are actually suffering from widespread developmental disorders. These children have communication barriers when interacting with others, and their characteristics are as follows:

- Poor social communication and barriers to interaction
- Limited cognitive ability, passive cognition
- Narrow interest, rigid repetition, lack of change and imagination
- Inherent habits and behaviors, difficulty to adapt to the new environment

孤独症儿童是目前常常被忽视的一类群体。他们有着类似性格内向的特征，实际却受着广泛性发育障碍的折磨。这类儿童在与人交往时存在着沟通障碍，其特点表现在：

- 社交交流差，互动障碍明显
- 认知能力有限，被动认知
- 兴趣狭窄，重复刻板，缺乏变化和想象力
- 固有行为习惯和行为方式，难以适应新环境

These symptoms can often be improved after training, but they cannot be completely cured. We should pay more attention to the needs of these groups and provide them with more care. At present, the commonly used teaching method is ABA application behavior analysis, which aims at the following two aspects of skills training:

- Perception
- Thinking

他们的这些症状在经过培训后往往可以得到改善，但却不能完全根治。我们应该更加关注这类群体的需求，为他们提供更多的关爱。目前常用的教学方法为ABA应用行为分析，针对以下两方面能力进行培养：

- 感知能力
- 思维运用能力

The specific training content includes:

- Item classification
- Commands to action
- Finding lost items
- Memorization
- Asking for help
- Using what has been learned 

具体训练内容有：

- 物品归类
- 指令动作
- 寻找丢失物
- 记事能力
- 请求帮助
- 利用所学

## Game Mechanics 游戏玩法

![ezgif-1-8de7005005e4](https://user-images.githubusercontent.com/60060750/128956571-dcd6b205-7784-4c19-9ba6-695b703f5ca1.gif)

(Watch the full [YouTube video](https://www.youtube.com/watch?v=LJjONGwVzpQ) 观看完整版[B站视频](https://www.bilibili.com/video/BV1hM4y157e3))

To stick with the universal design principle, we chose a cognitive improvement perspective that is also applicable to ordinary children's education, as follows:

1. Randomly draw from three basic figures (circle, square, triangle)
2. The user takes pictures of real objects that they think are similar to the received task and sends them back to the chatbot
3. After receiving the reply, the bot starts to judge and provides a feedback according to whether the picture matched the figure or not
4. The user chooses to end the game/play another round

出于对通用性设计的考虑，我们选择了同样适用于普通儿童教育的认知学提高角度进行以下设计：

1. 从三种基本图形（圆，正方，正三角）中随机抽取，机器人动图展示并以抽奖形式定格其中一种图形
2. 用户根据收到的图形拍取其认为相似的真实物体发送给机器人
3. 收到回复后机器人开始进行判断，根据图片匹配度提供回复
4. 用户选择结束游戏/再来一局

儿童根据所收获的星星可以兑换更多图形模板进行后续的匹配度游戏，也可以在社区与其他小朋友进行PK。

## Technical Documentation 技术文档

![process](https://media.licdn.cn/dms/image/C5612AQFLQ9C8NJbHMg/article-inline_image-shrink_1000_1488/0/1628340432970?e=1634169600&v=beta&t=hUZBR8LrcSo5d5zDXzQyYb9iei1TYwGIrn2_t84TNpM)

### Solution (Left-to-Right) 解决方案（从左到右）

1. The chatroom, where all the messaging happens back and forth between different users
2. Puppet service, an online service masking as an iPad device so it can login to the same Wechat account as your phone. After that, it has access to all of your chatrooms and private messages and can connect with the next layer, sending and receiving messages
3. Main Wechaty program that allows the program to access all the messages and attachments sent to the chatrooms and grabbed by the Puppet service. The main chat logic of the app will be built into the Wechaty part such as specific wake words, if-this-then-thats, all the rules for interaction basically
4. A standalone Python CV recognition service, using the Flask framework that the Wechaty program could call via a simple http API to transfer data back and forth

--

1. 微信聊天群或者小窗：正常互动（如：发送消息，接收消息，发/收图片等）
2. 微信Puppet代理：这个模块会拿你的微信账号登录到一台虚拟iPad设备上，获取到微信的聊天信息，再转到核心的聊天机器人模块
3. 核心Wechaty聊天机器人模块(C#)：机器人的所有操作逻辑在这里，比如游戏的流程和规则，以及跟图像识别模块的对接
4. 图像识别模块(Python)：因为图像识别跟Wechaty不是同一套语言实现的，把图像识别做成独立的服务，用http POST接口跟聊天机器人模块进行数据传送与对接

### Start-up Command 启动命令

#### Wechaty Module/模块

1. Run the C# code under the Wechaty directory
2. The code will generate a QR code, scan and log in to the Puppet proxy using your WeChat account

--

1. 运行Wechaty子目录之下的C#代码
2. 将使用移动设备的微信扫码登录程序显示的二维码，绑定Puppet代理

#### Computer Vision Recognition Module 图像识别模块

First, download the pretrained deep learning model on [Google Drive](https://drive.google.com/open?id=1PkGX9R-uTYpWBKX0lZRkE2qvvpz1-IiG) and save it to `CVModule/core/model`

首先，需要下载预训练好的深度学习模型：[Google Drive](https://drive.google.com/open?id=1PkGX9R-uTYpWBKX0lZRkE2qvvpz1-IiG) 或者 [Baidu NetDisk](https://pan.baidu.com/s/1Gm-YptzsVnHU0a6YkdjQaQ) (code: ilhx)，将文件保存到：`CVModule/core/model`

1. `cd CVModule`
2. `docker build . -t starchild`
3. `docker run -d --name starchild -p XXXX:1337 starchild`

Use the `dockerfile` under `CVModule` directory to generate a Docker image. When you run it, the service will be automatically started and listens to the port 1337. If you add a `-p XXXX:1337` flag in your `docker run` command, you can specify which port on the host machine will be mapped into the 1337 port within docker.

使用`CVModule`子目录下的`dockerfile`将代码封装成Docker镜像，再运行该镜像，服务会自动起来，监听容器里端口1337。可以通过容器启动的命令附加`-p XXXX:1337`将任何主机的端口转到容器里的1337端口。

_The image recognition algorithm and deep learning model were provided in the ["SCRN"-paper](https://github.com/wuzhe71/SCRN)._

_图像识别算法的深度学习模型是基于 ["SCRN"-paper](https://github.com/wuzhe71/SCRN) 实现的。_
