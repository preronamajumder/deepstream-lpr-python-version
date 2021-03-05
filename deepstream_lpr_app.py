#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import numpy as np
import cv2
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
from common.utils import long_to_int
import pyds
from threading import Thread
from queue import Queue
import os
import datetime

fps_streams={}

MUXER_OUTPUT_WIDTH=640
MUXER_OUTPUT_HEIGHT=480
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=640
TILED_OUTPUT_HEIGHT=480
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 0
pgie_classes_str= ["lpd"]

# tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.
def tiler_src_pad_buffer_probe(pad,info,u_data,args):

    frame_number=0
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    lp_dict = {}
    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        '''
        print("Frame Number is ", frame_meta.frame_num)
        print("Source id is ", frame_meta.source_id)
        print("Batch id is ", frame_meta.batch_id)
        print("Source Frame Width ", frame_meta.source_frame_width)
        print("Source Frame Height ", frame_meta.source_frame_height)
        print("Num object meta ", frame_meta.num_obj_meta)
        '''
        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta

        while l_obj is not None:
            try: 
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if args[3] == '0':
                #no ROI
                l_class = obj_meta.classifier_meta_list

                while l_class is not None:
                    try:
                        class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
                    except StopIteration:
                        break

                    l_label = class_meta.label_info_list

                    while l_label is not None:
                        try:
                            label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                        except StopIteration:
                            break

                        print(label_info.result_label)
                    
                        try:
                            l_label=l_label.next
                        except StopIteration:
                            break
                    try:
                        l_class=l_class.next
                    except StopIteration:
                        break
        
            else:
                l_user_meta = obj_meta.obj_user_meta_list
                while l_user_meta:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                        if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):             
                            user_meta_data = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                            print("Object {0} line crossing status: {1}".format(obj_meta.object_id, user_meta_data.lcStatus))
                            print("Object {0} roi status: {1}".format(obj_meta.object_id, user_meta_data.roiStatus))
                            # if user_meta_data.dirStatus: print("Object {0} moving in direction: {1}".format(obj_meta.object_id, user_meta_data.dirStatus))                    
                            # if user_meta_data.lcStatus: print("Object {0} line crossing status: {1}".format(obj_meta.object_id, user_meta_data.lcStatus))
                            # if user_meta_data.ocStatus: print("Object {0} overcrowding status: {1}".format(obj_meta.object_id, user_meta_data.ocStatus))
                            # if user_meta_data.roiStatus: print("Object {0} roi status: {1}".format(obj_meta.object_id, user_meta_data.roiStatus))

                            l_class = obj_meta.classifier_meta_list
                            # print(l_class)

                            if user_meta_data.roiStatus:
                                while l_class is not None:
                                    try:
                                        class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
                                    except StopIteration:
                                        break

                                    l_label = class_meta.label_info_list
                                    while l_label is not None:
                                        try:
                                            label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                                        except StopIteration:
                                            break

                                        print(label_info.result_label)

                                        try:
                                            l_label=l_label.next
                                        except StopIteration:
                                            break
                                    try:
                                        l_class=l_class.next
                                    except StopIteration:
                                        break

                    except StopIteration:
                        break

                try:
                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break

            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Get meta data from NvDsAnalyticsFrameMeta
        l_user = frame_meta.frame_user_meta_list
        while l_user:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META"):
                    user_meta_data = pyds.NvDsAnalyticsFrameMeta.cast(user_meta.user_meta_data)
                    # if user_meta_data.objInROIcnt: print("Objs in ROI: {0}".format(user_meta_data.objInROIcnt))                    
                    if user_meta_data.objLCCumCnt: print("Linecrossing Cumulative: {0}".format(user_meta_data.objLCCumCnt))
                    # if user_meta_data.objLCCurrCnt: print("Linecrossing Current Frame: {0}".format(user_meta_data.objLCCurrCnt))
                    # if user_meta_data.ocStatus: print("Overcrowding status: {0}".format(user_meta_data.ocStatus))
            except StopIteration:
                break
            try:
                l_user = l_user.next
            except StopIteration:
                break

        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def get_frame(gst_buffer, batch_id):
    n_frame=pyds.get_nvds_buf_surface(hash(gst_buffer),batch_id)
    #convert python array into numy array format.
    frame_image=np.array(n_frame,copy=True,order='C')
    #covert the array into cv2 default color format
    frame_image=cv2.cvtColor(frame_image,cv2.COLOR_RGBA2BGRA)
    return frame_image

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)   
    if name.find("nvv4l2decoder") != -1:
        if is_aarch64():
            print("Seting bufapi_version\n")
            Object.set_property("bufapi-version",True)

def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):

    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("Usage: {} [1:us model|2: ch_model] [1:filesink|2:fakesink|"
        "3:display sink] [0:ROI disable|0:ROI enable] <In mp4 filename> <in mp4 filename> ... "
        "<out H264 filename>\n".format(args[0]))
        sys.exit(1)

    print(args)
    for i in range(0,len(args)-5):
        fps_streams["stream{0}".format(i)]=GETFPS(i)
    number_sources=len(args)-5
    print(number_sources)
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=args[i+4]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        else:
            uri_name="file://"+uri_name
        print(uri_name)
        print("*********************")
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    queue8=Gst.ElementFactory.make("queue","queue8")
    queue9=Gst.ElementFactory.make("queue","queue9")
    queue10=Gst.ElementFactory.make("queue","queue10")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    pipeline.add(queue8)
    pipeline.add(queue9)
    pipeline.add(queue10)

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating tiler \n ")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    print("Creating nvdsanalytics \n ")
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    if not nvanalytics:
        sys.stderr.write(" Unable to create nvanalytics \n")
    nvanalytics.set_property("config-file", "config_nvdsanalytics.txt")

    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie1:
        sys.stderr.write(" Unable to make sgie1 \n")

    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    if not sgie2:
        sys.stderr.write(" Unable to make sgie2 \n")

    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")

    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)

    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    # nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    # nvosd.set_property('display-text',OSD_DISPLAY_TEXT)

    if(is_aarch64()):
        print("Creating transform \n ")
        transform=Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    if args[2] == '1':
        print("Creating FileSink \n")
        sink = Gst.ElementFactory.make("filesink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create file sink \n")
        sink.set_property("location", args[-1])
    elif args[2] == '3':
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")
    elif args[2] == '2':
        print("Creating FakeSink \n")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        if not sink:
            sys.stderr.write(" Unable to create fake sink \n")
    else:
        print(" Invalid argument for sink \n")
        sys.exit(1)

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)

    pgie.set_property('config-file-path', "trafficamnet_config.txt")
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
        pgie.set_property("batch-size",number_sources)

    # Set properties of tracker
    config = configparser.ConfigParser()
    config.read('lpr_sample_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process':
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process',
                                 tracker_enable_batch_process)

    if args[1] == '1':
        sgie1.set_property('config-file-path', "lpd_us_config.txt")
    elif args[1] == '2':
        sgie1.set_property('config-file-path', "lpd_ccpd_config.txt")
    else:
        print(" Invalid argument for LP detector model \n")
        sys.exit(1)
    sgie1.set_property('process-mode', 2)

    if args[1] == '1':
        sgie2.set_property('config-file-path', "lpr_config_sgie_us.txt")
    elif args[1] == '2':
        sgie2.set_property('config-file-path', "lpr_config_sgie_ch.txt")
    else:
        print(" Invalid argument for LP recognition model \n")
        sys.exit(1)
    sgie2.set_property('process-mode', 2)

    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)
    sink.set_property("sync",0)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")
    
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvanalytics)
    pipeline.add(sgie1)
    pipeline.add(sgie2)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    if args[3] == '0':            #ROI disable
        streammux.link(pgie)
        pgie.link(queue1)
        queue1.link(tracker)
        tracker.link(queue2)
        queue2.link(sgie1)
        sgie1.link(queue3)
        queue3.link(sgie2)
        sgie2.link(queue4)
        queue4.link(nvvidconv1)
        nvvidconv1.link(queue5)
        queue5.link(filter1)
        filter1.link(queue6)
        queue6.link(tiler)
        tiler.link(queue7)
        queue7.link(nvvidconv)
        nvvidconv.link(queue8)
        queue8.link(nvosd)
        if is_aarch64():
            nvosd.link(queue9)
            queue9.link(transform)
            transform.link(sink)
        else:
            nvosd.link(queue9)
            queue9.link(sink)

    elif args[3] == '1':               #ROI enable
        streammux.link(pgie)
        pgie.link(queue1)
        queue1.link(tracker)
        tracker.link(queue2)
        queue2.link(nvanalytics)
        nvanalytics.link(queue3)
        queue3.link(sgie1)
        sgie1.link(queue4)
        queue4.link(sgie2)
        sgie2.link(queue5)
        queue5.link(nvvidconv1)
        nvvidconv1.link(queue6)
        queue6.link(filter1)
        filter1.link(queue7)
        queue7.link(tiler)
        tiler.link(queue8)
        queue8.link(nvvidconv)
        nvvidconv.link(queue9)
        queue9.link(nvosd)
        if is_aarch64():
            nvosd.link(queue10)
            queue10.link(transform)
            transform.link(sink)
        else:
            nvosd.link(queue10)
            queue10.link(sink) 
    else:
        print(" Invalid argument for ROI enable \n")
        sys.exit(1)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    tiler_src_pad=tiler.get_static_pad("sink")
    if not tiler_src_pad:
        sys.stderr.write(" Unable to get sink pad \n")
    else:
        tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0, args)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if ((len(args) - 1) > i > 3):
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    thread_stop = True
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

