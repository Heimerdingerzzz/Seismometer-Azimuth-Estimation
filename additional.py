ble, bln,bhe,bhn,nbhe,nbhn=BLE.data, BLN.data, BHE.data, BHN.data, new_bhe.data,new_bhn.data
fble = BLE.filter("bandpass",freqmin=0.1,freqmax=1,zerophase=True).data
fbln = BLN.filter("bandpass",freqmin=0.1,freqmax=1,zerophase=True).data
fnbhe = new_bhe.filter("bandpass",freqmin=0.1,freqmax=1,zerophase=True).data
fnbhn = new_bhn.filter("bandpass",freqmin=0.1,freqmax=1,zerophase=True).data  
df = pd.DataFrame(data = np.corrcoef([ble,bln,bhe,bhn,nbhe,nbhn,fble,fbln,fnbhe,fnbhn]), \
              columns= ["BLE", "BLN", "BHE", "BHN", "new_BHE","new_BHN","fble","fbln","fnew_BHE","fnew_BHN"], \
              index = ["BLE", "BLN", "BHE", "BHN", "new_BHE","new_BHN","fble","fbln","fnew_BHE","fnew_BHN"] )
df.to_csv(r"/home/anotherone/文档/mycode/artwork/corrcoef.csv")
