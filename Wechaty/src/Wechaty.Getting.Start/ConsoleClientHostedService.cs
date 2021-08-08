using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Newtonsoft.Json;
using TencentCloud.Common;
using TencentCloud.Common.Profile;
using TencentCloud.Nlp.V20190408;
using TencentCloud.Nlp.V20190408.Models;
using Wechaty.Module.Filebox;
using Wechaty.Plugin;
using Wechaty.User;
using Wechaty_Child.Models;

namespace Wechaty_Child
{
    public class ConsoleClientHostedService : IHostedService
    {
        private static HttpClient _client = new HttpClient();
        private static int _lastIndex = 0;
        private static Dictionary<string, Question> _questions = new Dictionary<string, Question>();
        private static readonly string[] _sharps = new string[] { "round", "square", "triangle" };
        private static readonly string[] _simplesharps = new string[] { "r", "s", "t" };
        private readonly IConfiguration _configuration;

        public ConsoleClientHostedService(IConfiguration configuration)
        {
            _configuration = configuration;
        }

        public void ConfigureService(IServiceCollection services)
        {


        }


        private static Wechaty.Wechaty bot;

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            var PuppetOptions = new Wechaty.Module.Puppet.Schemas.PuppetOptions()
            {
                Token = "puppet_paimon_e6389840ba12c23ab01ef3b4a575e77d",
            };
            bot = new Wechaty.Wechaty(PuppetOptions);

            _client.BaseAddress = new Uri("http://localhost:1337");

            // Automatic plug-in registration
            //var serviceCollection = new ServiceCollection()
            //    .AddSingleton<IWechatPlugin, ScanPlugin>()
            //    .AddSingleton<IWechatPlugin, DingDongPlugin>();
            //var plugins = serviceCollection.BuildServiceProvider().GetServices<IWechatPlugin>().ToArray();


            // Manual plug-in registration
            var qrCodeTerminalPlugin = new QRCodeTerminalPlugin();
            var dingDongPlugin = new DingDongPlugin();
            bot.Use(qrCodeTerminalPlugin)
               .Use(dingDongPlugin);



            await bot
              //.OnScan(WechatyScanEventListener)
              //.OnLogin(async (ContactSelf user) =>
              //{
              //    //Console.WriteLine($"{user.Name}在{DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")}上线了！");
              //})
              .OnMessage(WechatyMessageEventListenerAsync)
              .OnHeartbeat(WechatyHeartbeatEventListener)
              .OnRoomInvite(WechatyRoomInviteEventListener)
              .OnRoomJoin(WechatyRoomJoinEventListener)
              .OnRoomLeave(WechatyRoomLeaveEventListener)
              .OnRoomTopic(WechatyRoomTopicEventListener)
              .Start();

            LoopQuestionsQUeue();
        }

        public static void WechatyLoginEventListener(ContactSelf user)
        {
            //Console.WriteLine($"{user.Name}在{DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")}上线了！");
        }

        private static void WechatyHeartbeatEventListener(object data)
        {
            //Console.WriteLine(JsonConvert.SerializeObject(data));
        }

        //private static void WechatyScanEventListener(string qrcode, ScanStatus status, string? data)
        //{
        //    Console.WriteLine(qrcode);
        //    const string QrcodeServerUrl = "https://wechaty.github.io/qrcode/";
        //    if (status == ScanStatus.Waiting || status == ScanStatus.Timeout)
        //    {
        //        var qrcodeImageUrl = QrcodeServerUrl + qrcode;
        //        Console.WriteLine(qrcodeImageUrl);
        //    }
        //    else if (status == ScanStatus.Scanned)
        //    {
        //        Console.WriteLine(data);
        //    }
        //}

        private static void LoopQuestionsQUeue()
        {
            var _ = Task.Run(async () =>
              {
                  while (true)
                  {
                      foreach (var item in _questions)
                      {
                          if (item.Value != null)
                          {
                              if (DateTime.Now.AddMinutes(-3) >= item.Value.RefreshTime )
                              {
                                  await bot.Say(item.Value.ConversationId, "欢乐的时光总是短暂～很高兴遇见你们！欢迎再来找我玩，输入“开始游戏”我就会出现啦！");
                                  _questions.Remove(item.Key);
                              }
                              else if (DateTime.Now.AddMinutes(-1) >= item.Value.RefreshTime && !item.Value.OuttimeWarning)
                              {
                                  await bot.Say(item.Value.ConversationId, "还没有找到想要的图形嘛？若三分钟内未发送图片视为自动放弃哦～");
                                  item.Value.OuttimeWarning = true;
                                  
                              }


                          }
                      }

                      await Task.Delay(1000);
                  }
              });
        }

        private static async void WechatyMessageEventListenerAsync(Message message)
        {
            try
            {
                if (message.Text == "结束游戏")
                {
                    _questions.Remove(message.Coversation.Id);
                }

                if (message.Text == "开始游戏")
                {
                    //随机挑选一张图
                    var index = new Random().Next(3);

                    var file = $"{AppDomain.CurrentDomain.BaseDirectory}Scripts\\starChild\\{_sharps[index]}.png";
                    if (!File.Exists(file))
                        return;

                    await Task.Delay(50);
                    await bot.Say(message.Coversation.Id, FileBox.FromBase64(ImageToBase64(file), Path.GetFileName(file)));

                    var question = new Question()
                    {
                        SimpleKey = _simplesharps[index],
                        ConversationId = message.Coversation.Id
                    };
                    if (_questions.ContainsKey(message.Coversation.Id))
                    {
                        _questions[message.Coversation.Id] = question;
                    }
                    else
                    {
                        _questions.Add(message.Coversation.Id, question);
                    }

                    _lastIndex = index;
                }

                if (message.Type == Wechaty.Module.Puppet.Schemas.MessageType.Image
                    && _questions.ContainsKey(message.Coversation.Id)
                    && _questions[message.Coversation.Id] != null)
                {

                    var bytes = await bot.GetFile(message.Id);
                    var base64 = Convert.ToBase64String(bytes);
                    var content = new StringContent(JsonConvert.SerializeObject(new
                    {
                        gt = _simplesharps[_lastIndex],
                        img_b64 = base64
                    }));

                    content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");

                    var resp = await _client.PostAsync("api/getPrediction", content);
                    if (resp.IsSuccessStatusCode)
                    {
                        var result = await resp.Content.ReadAsStringAsync();
                        var obj = JsonConvert.DeserializeObject<dynamic>(result);
                        if (obj.status == 0)
                        {
                            await bot.Say(message.Coversation.Id, ((bool)obj.message) ? $"恭喜{message.From?.Name}同学" : $"{message.From?.Name}同学还差一点点，继续加油呀！");
                        }
                    }

                    _questions[message.Coversation.Id].RefreshTime = DateTime.Now;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }
        }

        public static void Base64ToImage(string base64)
        {
            base64 = base64.Replace("data:image/png;base64,", "").Replace("data:image/jgp;base64,", "").Replace("data:image/jpg;base64,", "").Replace("data:image/jpeg;base64,", "");//将base64头部信息替换
            byte[] bytes = Convert.FromBase64String(base64);
            MemoryStream memStream = new MemoryStream(bytes);
            System.Drawing.Image mImage = System.Drawing.Image.FromStream(memStream);
            Bitmap bp = new Bitmap(mImage);
            bp.Save("C:/Users/" + DateTime.Now.ToString("yyyyMMddHHss") + ".jpg", System.Drawing.Imaging.ImageFormat.Jpeg);//注意保存路径
        }

        /// <summary>
        /// Image 转成 base64
        /// </summary>
        /// <param name="fileFullName"></param>
        public static string ImageToBase64(string fileFullName)
        {
            try
            {
                Bitmap bmp = new Bitmap(fileFullName);
                MemoryStream ms = new MemoryStream();
                bmp.Save(ms, System.Drawing.Imaging.ImageFormat.Jpeg);
                byte[] arr = new byte[ms.Length];
                ms.Position = 0;
                ms.Read(arr, 0, (int)ms.Length);
                ms.Close();
                return Convert.ToBase64String(arr);
            }
            catch (Exception ex)
            {
                return null;
            }
        }

        #region Room
        private static void WechatyRoomInviteEventListener(RoomInvitation roomInvitation)
        {
            // 1102977037

        }

        private static void WechatyRoomJoinEventListener(Room room, IReadOnlyList<Contact> inviteeList, Contact inviter, DateTime? date)
        {
            Console.WriteLine($"{inviter.Name} invites {string.Join(",", inviteeList.Select(x => x.Name))} into {room.GetTopic().Result} room !");
        }

        private static void WechatyRoomLeaveEventListener(Room room, IReadOnlyList<Contact> leaverList, Contact remover, DateTime? date)
        {

        }

        private static void WechatyRoomTopicEventListener(Room room, string newTopic, string oldTopic, Contact changer, DateTime? date)
        {
            Console.WriteLine($"{changer.Name} update room topic as {newTopic}");
        }
        #endregion

        /// <summary>
        /// Stop
        /// </summary>
        /// <param name="cancellationToken"></param>
        /// <returns></returns>
        public async Task StopAsync(CancellationToken cancellationToken)
        {
            Process.GetCurrentProcess().Kill();
            return;
        }
    }
}
