FasdUAS 1.101.10   ��   ��    k             l      ��  ��   WQ********************************************
Record a Single `QuickTime` Movie
Args:
    1. name: The name of the movie.
    2. seconds: The length of the movie you want to record in seconds.
Usage:
    > osascript record_sceen 'name.mov' 5
    > osascript record_sceen <file_name> <seconds>
*********************************************     � 	 	� * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 R e c o r d   a   S i n g l e   ` Q u i c k T i m e `   M o v i e 
 A r g s : 
         1 .   n a m e :   T h e   n a m e   o f   t h e   m o v i e . 
         2 .   s e c o n d s :   T h e   l e n g t h   o f   t h e   m o v i e   y o u   w a n t   t o   r e c o r d   i n   s e c o n d s . 
 U s a g e : 
         >   o s a s c r i p t   r e c o r d _ s c e e n   ' n a m e . m o v '   5 
         >   o s a s c r i p t   r e c o r d _ s c e e n   < f i l e _ n a m e >   < s e c o n d s > 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *   
  
 l     ��������  ��  ��        l     ��  ��    ~ xhttp://stackoverflow.com/questions/17328782/applescript-quicktime-start-and-stop-a-movie-recording-and-export-it-to-disk     �   � h t t p : / / s t a c k o v e r f l o w . c o m / q u e s t i o n s / 1 7 3 2 8 7 8 2 / a p p l e s c r i p t - q u i c k t i m e - s t a r t - a n d - s t o p - a - m o v i e - r e c o r d i n g - a n d - e x p o r t - i t - t o - d i s k   ��  i         I     �� ��
�� .aevtoappnull  �   � ****  o      ���� 0 argv  ��    k     ]       r         n         4    �� 
�� 
cobj  m    ����   o     ���� 0 argv    o      ���� 0 	moviename 	movieName      r         n     ! " ! 4    �� #
�� 
cobj # m   	 
����  " o    ���� 0 argv     o      ���� 0 delayseconds delaySeconds   $ % $ r     & ' & b     ( ) ( l    *���� * I   �� + ,
�� .earsffdralis        afdr + m    ��
�� afdrdesk , �� -��
�� 
rtyp - m    ��
�� 
ctxt��  ��  ��   ) o    ���� 0 	moviename 	movieName ' o      ���� 0 filepath filePath %  . / . r    ! 0 1 0 N     2 2 4    �� 3
�� 
file 3 o    ���� 0 filepath filePath 1 o      ���� 0 f   /  4 5 4 l  " "�� 6 7��   6 H Bset windowID to id of first window whose name = "Screen Recording"    7 � 8 8 � s e t   w i n d o w I D   t o   i d   o f   f i r s t   w i n d o w   w h o s e   n a m e   =   " S c r e e n   R e c o r d i n g " 5  9 : 9 l  " "��������  ��  ��   :  ;�� ; O   " ] < = < k   & \ > >  ? @ ? r   & - A B A I  & +������
�� .MVWRnscrnull��� ��� null��  ��   B o      ���� (0 newscreenrecording newScreenRecording @  C�� C O   . \ D E D k   2 [ F F  G H G I  2 7������
�� .MVWRstarnull���     docu��  ��   H  I J I I  8 =�� K��
�� .sysodelanull��� ��� nmbr K o   8 9���� 0 delayseconds delaySeconds��   J  L M L I  > C������
�� .MVWRpausnull���     docu��  ��   M  N O N I  D M�� P Q
�� .coresavenull���     obj  P o   D E���� (0 newscreenrecording newScreenRecording Q �� R��
�� 
kfil R o   H I���� 0 f  ��   O  S T S I  N S������
�� .MVWRstopnull���     docu��  ��   T  U V U I  T Y�� W��
�� .coreclosnull���     obj  W o   T U���� (0 newscreenrecording newScreenRecording��   V  X Y X l  Z Z��������  ��  ��   Y  Z [ Z l  Z Z�� \ ]��   \ m gset newScreenRecordingDoc to first document whose name = (get name of first window whose id = windowID)    ] � ^ ^ � s e t   n e w S c r e e n R e c o r d i n g D o c   t o   f i r s t   d o c u m e n t   w h o s e   n a m e   =   ( g e t   n a m e   o f   f i r s t   w i n d o w   w h o s e   i d   =   w i n d o w I D ) [  _�� _ l  Z Z�� ` a��   ` S Mtell newScreenRecordingDoc to export in filePath using settings preset "iPod"    a � b b � t e l l   n e w S c r e e n R e c o r d i n g D o c   t o   e x p o r t   i n   f i l e P a t h   u s i n g   s e t t i n g s   p r e s e t   " i P o d "��   E o   . /���� (0 newscreenrecording newScreenRecording��   = m   " # c c�                                                                                  mgvr  alis    v  Macintosh HD               Ηs�H+  r�QuickTime Player.app                                           
���5L�        ����  	                Applications    Η��      �5��    r�  /Macintosh HD:Applications: QuickTime Player.app   *  Q u i c k T i m e   P l a y e r . a p p    M a c i n t o s h   H D  !Applications/QuickTime Player.app   / ��  ��  ��       �� d e��   d ��
�� .aevtoappnull  �   � **** e �� ���� f g��
�� .aevtoappnull  �   � ****�� 0 argv  ��   f ���� 0 argv   g �������������������� c������������������
�� 
cobj�� 0 	moviename 	movieName�� 0 delayseconds delaySeconds
�� afdrdesk
�� 
rtyp
�� 
ctxt
�� .earsffdralis        afdr�� 0 filepath filePath
�� 
file�� 0 f  
�� .MVWRnscrnull��� ��� null�� (0 newscreenrecording newScreenRecording
�� .MVWRstarnull���     docu
�� .sysodelanull��� ��� nmbr
�� .MVWRpausnull���     docu
�� 
kfil
�� .coresavenull���     obj 
�� .MVWRstopnull���     docu
�� .coreclosnull���     obj �� ^��k/E�O��l/E�O���l �%E�O*��/E�O� 8*j E�O� +*j O�j O*j O�a �l O*j O�j OPUU ascr  ��ޭ