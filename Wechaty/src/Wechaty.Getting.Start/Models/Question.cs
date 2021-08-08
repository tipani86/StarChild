using System;
using System.Collections.Generic;
using System.Text;

namespace Wechaty_Child.Models
{
    public class Question
    { 
        public DateTime RefreshTime { get; set; } = DateTime.Now;

        public string SimpleKey { get; set; }

        public string ConversationId { get; set; }

        public bool OuttimeWarning { get; set; }
    }
}
