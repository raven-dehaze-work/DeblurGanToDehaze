package com.raven.dehazedemo;

import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;

/**
 * 上传图片类
 */
public class UpLoadImgModule {

    private final String IP = "103.46.128.45";
    private final int PORT= 13204;


    /**
     * 上传回调接口
     */
    public interface CallBackListener{

        /**
         * 去雾成功后回调
         * @param dehazeImgPath
         */
        void onDehazeSuccess(String dehazeImgPath);

        /**
         * 中途出错，回调这个函数
         */
        void onDehazeError(String msg);
    }

    private CallBackListener callBackListener;

    public UpLoadImgModule(CallBackListener callBackListener) {
        this.callBackListener = callBackListener;
    }

    /**
     * 上传图片
     * @param filePath
     */
    public void uploadImg(String filePath){
        try{
            new SocketThread(filePath).start();
        }catch (Exception e){
            e.printStackTrace();
            callBackListener.onDehazeError("网络线程发生错误");
        }

    }

    /**
     * Socket通信线程，主要实现功能:
     * 1. 发送试卷正反面到服务端
     * 2. 发送完成后建立接受分析结果socket
     */
    private class SocketThread extends Thread {

        private String path;
        Socket socket;

        SocketThread(String imgPath) {
            this.path = imgPath;
        }

        /**
         * 发送图片
         * @param path
         * @throws IOException
         */
        private void sendImg(String path) throws IOException {
            // 创建一个Socket对象，并指定服务端的IP及端口号
            socket = new Socket(IP, PORT);
            InputStream inputStream = new FileInputStream(path);
            // 获取Socket的OutputStream对象用于发送数据。
            OutputStream outputStream = socket.getOutputStream();
            // 创建一个byte类型的buffer字节数组，用于存放读取的本地文件
            byte buffer[] = new byte[10 * 1024];
            int temp = 0;
            // 循环读取文件
            while ((temp = inputStream.read(buffer)) != -1) {
                // 把数据写入到OuputStream对象中
                outputStream.write(buffer, 0, temp);
                // 发送读取的数据到服务端
                outputStream.flush();
            }
            socket.close();
        }


        /**
         * 接收来自服务器去雾后的图片
         * @return 将去雾图保存在本地的本地路径
         * @throws IOException
         */
        private String recieveImg() throws IOException {
            // 发送完成，等待结果
            socket = new Socket(IP, PORT);
            InputStream inputStream = socket.getInputStream();
            // 保存图片
            // 获取雾图图片文件名和目录
            File hazeFile = new File(this.path);
            String fileName = hazeFile.getName().substring(0,hazeFile.getName().indexOf('.'));
            Log.i("file name",fileName);
            File dehazeImg = new File(new File("/sdcard/DehazeDemo/DehazeImgs"),fileName+"_dehaze.jpg");
            //创建图片字节流
            FileOutputStream fos = new FileOutputStream(dehazeImg);
            byte[] buf = new byte[1024];
            int len = 0;
            //往字节流里写图片数据
            while ((len = inputStream.read(buf)) != -1)
            {
                fos.write(buf,0,len);
            }
            return dehazeImg.getAbsolutePath();
        }

        @Override
        public void run() {
            super.run();
            try {
                sendImg(this.path);
                Log.i(MainActivity.class.getSimpleName(),"finish img 1");
                String dehazeImgPath = recieveImg();
                callBackListener.onDehazeSuccess(dehazeImgPath);
            } catch (IOException e) {
                e.printStackTrace();
                Log.e(MainActivity.class.getSimpleName(),"连接失败");
                callBackListener.onDehazeError("Socket 连接失败");
            }
        }


    }

}
