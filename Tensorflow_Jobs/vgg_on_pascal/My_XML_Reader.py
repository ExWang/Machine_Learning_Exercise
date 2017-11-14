# -*- coding: UTF-8 -*-

from xml.etree import ElementTree as et
import myPath


def read_xml(text):
    """''读xml文件"""
    # 加载XML文件（2种方法,一是加载指定字符串，二是加载指定文件）
    # root = ElementTree.parse(r"D:/test.xml")
    xmldict = {}
    root = et.fromstring(text)

    xml_folder = root.find('folder')
    xmldict['folder'] = xml_folder.text

    xml_filename = root.find('filename')
    xmldict['filename'] = xml_filename.text

    xml_source = root.find('source')
    temp_dict = {}
    for one_sum in xml_source:
        temp_dict[one_sum.tag] = one_sum.text
    xmldict['source'] = temp_dict

    xml_size = root.find('size')
    temp_dict = {}
    for one_sum in xml_size:
        temp_dict[one_sum.tag] = one_sum.text
    xmldict['size'] = temp_dict

    xml_segmented = root.find('segmented')
    xmldict['segmented'] = xml_segmented.text

    xml_all_object = root.findall("object")
    object_num = 0
    temp_dict2 = {}
    for one_object in xml_all_object:
        temp_dict3 = {}
        for one_sum in one_object:
            if one_sum.tag == 'bndbox':
                temp_dict4 = {}
                for bbox_coord in one_sum:
                    temp_dict4[bbox_coord.tag] = bbox_coord.text
                temp_dict3[one_sum.tag] = temp_dict4
            elif one_sum.tag == 'actions':
                temp_dict4 = {}
                for bbox_coord in one_sum:
                    temp_dict4[bbox_coord.tag] = bbox_coord.text
                temp_dict3[one_sum.tag] = temp_dict4
            elif one_sum.tag == 'point':
                temp_dict4 = {}
                for bbox_coord in one_sum:
                    temp_dict4[bbox_coord.tag] = bbox_coord.text
                temp_dict3[one_sum.tag] = temp_dict4
            else:
                temp_dict3[one_sum.tag] = one_sum.text
        object_name = 'object_'+str(object_num)
        temp_dict2[object_name] = temp_dict3
        object_num += 1
    xmldict['object'] = temp_dict2

    return xmldict


def read_xml2dict(xmlpath):
    xml_dict = read_xml(open(xmlpath).read())
    return xml_dict



