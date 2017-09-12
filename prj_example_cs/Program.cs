using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;   //using dll
//use x64
namespace prj_example_cs
{
    class Program
    {
#if DEBUG
        [DllImport("YOLOv2SEd.dll", EntryPoint = "YoloLoad", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr YoloLoad(string cfg, string weights);
        [DllImport("YOLOv2SEd.dll", EntryPoint = "YoloTrain", CallingConvention = CallingConvention.Cdecl)]
        public static extern void YoloTrain(string _base_dir, string _datafile, string _cfgfile);
        [DllImport("YOLOv2SEd.dll", EntryPoint = "YoloDetectFromFile", CallingConvention = CallingConvention.Cdecl)]
        public static extern int YoloDetectFromFile(string img_path, IntPtr _net, float threshold, float[] result, int result_sz);
#else
        [DllImport("YOLOv2SE.dll", EntryPoint = "YoloLoad", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr YoloLoad(string cfg, string weights);
        [DllImport("YOLOv2SE.dll", EntryPoint = "YoloTrain", CallingConvention = CallingConvention.Cdecl)]
        public static extern void YoloTrain(string _base_dir, string _datafile, string _cfgfile);
        [DllImport("YOLOv2SE.dll", EntryPoint = "YoloDetectFromFile", CallingConvention = CallingConvention.Cdecl)]
        public static extern int YoloDetectFromFile(string img_path, IntPtr _net, float threshold, float[] result, int result_sz);
#endif
        static void Main(string[] args)
        {
            IntPtr network = YoloLoad("../../network/yolo.cfg", "../../network/yolo.weights");
            float[] result = new float[2048];
            int sz = YoloDetectFromFile("../../test.jpg", network, 0.09F, result, 2048);
            for (int i = 0; i < sz; i++)
            {
                int kind = (int)result[i * 6 + 0];
                float cval = result[i * 6 + 1];
                float left = result[i * 6 + 2];
                float top = result[i * 6 + 3];
                float width = result[i * 6 + 4];
                float height = result[i * 6 + 5];
                Console.WriteLine(kind);
                Console.WriteLine(cval);
                Console.WriteLine(left);
                Console.WriteLine(top);
                Console.WriteLine(width);
                Console.WriteLine(height);
            }
        }
    }
}
