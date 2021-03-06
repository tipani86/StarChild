using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using github.wechaty.grpc.puppet;
using Google.Protobuf;
using Newtonsoft.Json;
using Wechaty.Module.Filebox;
using Wechaty.Module.Puppet.Schemas;

namespace Wechaty.Module.PuppetService
{
    public partial class GrpcPuppet
    {
        #region Message
        public override async Task<string> MessageContact(string messageId)
        {
            var request = new MessageContactRequest
            {
                Id = messageId
            };

            var response = await grpcClient.MessageContactAsync(request);
            return response.Id;
        }

        public override async Task<FileBox> MessageFile(string messageId)
        {
            var request = new MessageFileRequest
            {
                Id = messageId
            };

            var response = await grpcClient.MessageFileAsync(request);
            var filebox = response.Filebox;
            return FileBox.FromJson(filebox);

        }

        public override async Task<FileBox> MessageImage(string messageId, Puppet.Schemas.ImageType imageType)
        {
            var request = new MessageImageRequest
            {
                Id = messageId,
                Type = (github.wechaty.grpc.puppet.ImageType)imageType
            };

            var response = await grpcClient.MessageImageAsync(request);
            var fileBox = response.Filebox;
            return FileBox.FromJson(fileBox);
        }

        public override async Task<byte[]> MessageImageStream(string messageId, Puppet.Schemas.ImageType imageType)
        {
            var request = new MessageImageStreamRequest
            {
                Id = messageId,
                Type = (github.wechaty.grpc.puppet.ImageType)imageType
            };

            var response = grpcClient.MessageImageStream(request);
            var resp = new MessageImageStreamResponse();
            var bytes = new List<byte>();
            while (await response.ResponseStream.MoveNext(System.Threading.CancellationToken.None))
            {
                bytes.AddRange(response.ResponseStream.Current.FileBoxChunk.Data.ToByteArray());
            }

            return bytes.ToArray();
        }

        public override async Task<MiniProgramPayload> MessageMiniProgram(string messageId)
        {
            var request = new MessageMiniProgramRequest
            {
                Id = messageId
            };

            var response = await grpcClient.MessageMiniProgramAsync(request);
            var payload = JsonConvert.DeserializeObject<MiniProgramPayload>(response.MiniProgram);
            return payload;
        }

        public override async Task<bool> MessageRecall(string messageId)
        {
            var request = new MessageRecallRequest
            {
                Id = messageId
            };

            var response = await grpcClient.MessageRecallAsync(request);
            if (response == null)
            {
                return false;
            }
            return response.Success;
        }

        public override async Task<string?> MessageSendContact(string conversationId, string contactId)
        {
            var request = new MessageSendContactRequest()
            {
                ConversationId = conversationId,
                ContactId = contactId
            };

            var response = await grpcClient.MessageSendContactAsync(request);
            return response?.Id;
        }

        public override async Task<string?> MessageSendFile(string conversationId, FileBox file)
        {
            var request = new MessageSendFileRequest
            {
                ConversationId = conversationId,
                Filebox = JsonConvert.SerializeObject(file.ToJson())
            };

            var response = await grpcClient.MessageSendFileAsync(request);
            return response?.Id;
        }

        public override async Task<string?> MessageSendMiniProgram(string conversationId, MiniProgramPayload miniProgramPayload)
        {
            var request = new MessageSendMiniProgramRequest
            {
                ConversationId = conversationId,
                MiniProgram = JsonConvert.SerializeObject(miniProgramPayload)
            };

            var response = await grpcClient.MessageSendMiniProgramAsync(request);
            return response?.Id;
        }

        public override Task<string?> MessageSendText(string conversationId, string text, params string[]? mentionIdList)
        {
            var request = new MessageSendTextRequest()
            {
                ConversationId = conversationId,
                Text = text,
                //MentonalIds = mentonalIds
            };

            var response = grpcClient.MessageSendText(request);
            return Task.FromResult(response?.Id);
        }

        public override async Task<string?> MessageSendText(string conversationId, string text, IEnumerable<string>? mentionIdList)
        {
            var request = new MessageSendTextRequest()
            {
                ConversationId = conversationId,
                Text = text,
                //MentonalIds = mentonalIds
            };

            var response = await grpcClient.MessageSendTextAsync(request);
            return response?.Id;
        }

        public override async Task<string?> MessageSendUrl(string conversationId, UrlLinkPayload urlLinkPayload)
        {
            var request = new MessageSendUrlRequest()
            {
                ConversationId = conversationId,
                UrlLink = JsonConvert.SerializeObject(urlLinkPayload)
            };

            var response = await grpcClient.MessageSendUrlAsync(request);
            return response?.Id;
        }

        public override async Task<UrlLinkPayload> MessageUrl(string messageId)
        {
            var request = new MessageUrlRequest()
            {
                Id = messageId
            };

            var response = await grpcClient.MessageUrlAsync(request);
            var payload = JsonConvert.DeserializeObject<UrlLinkPayload>(response.UrlLink);
            return payload;
        }
        #endregion
    }
}
